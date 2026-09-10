# Part 06 — `simulation.{h,cpp}`, part 1: the setup

Files: `src/simulation.h` (131 lines), `src/simulation.cpp` (499 lines).
This part covers the header and everything up to (but not including)
`runBatch`. Part 07 covers `runBatch` and the rest.

**Job of `Simulation`:** own all the GPU resources for the N-body sim
(buffers, pipelines, descriptors, query pool, a reusable command
buffer/fence), and provide `runBatch` to advance the simulation while
recording timings.

---

## 1. The header structs

### `Vec3` (lines 10-12)

```cpp
struct Vec3 { float x = 0, y = 0, z = 0; };
```

A tiny 3-float vector for CPU-side math (the GPU has `vec3` built in; the CPU
doesn't). Used for accelerations and momentum in the tests.

### `Particle` (lines 16-19)

```cpp
struct Particle {
    float px = 0, py = 0, pz = 0, mass = 0;
    float vx = 0, vy = 0, vz = 0, _pad = 0;
};
```

Eight floats = 32 bytes, laid out to match the shader's two `vec4`s under
`std430` (see Part 04 §4). `_pad` is a deliberate unused float so the
struct is exactly 32 bytes.

### `SimParams` (lines 21-26)

```cpp
struct SimParams {
    uint32_t particleCount = 0;
    float gravity = 1.0f;
    float softening = 0.5f;
    float dt = 0.005f;
};
```

The knobs. `particleCount` is set per run; the other three have physics
defaults. These get copied into a push-constant block each step.

### The two constants (lines 30-31)

```cpp
static constexpr uint32_t kTimestampsPerStep = 6;
static constexpr uint32_t kMaxBatch = 64;
```

6 timestamps per step (before force, after force, after barrier1, after
integrate, after barrier2, after copy). Largest batch of steps the code will
record into one command buffer. The query pool is sized `6 * 64 = 384`.

### `StepTimings` (lines 35-50)

```cpp
struct StepTimings {
    bool gpuValid = false;
    uint64_t rawTicks[6] = {};
    double timestampPeriodNs = 1.0;

    double forceDispatchNs = 0;     // ticks[0]->[1]
    double barrier1Ns = 0;          // ticks[1]->[2]
    double integrateDispatchNs = 0; // ticks[2]->[3]
    double barrier2Ns = 0;          // ticks[3]->[4]
    double copyNs = 0;              // ticks[4]->[5]
    double gpuTotalNs = 0;          // ticks[0]->[5]

    double submitCallNs = 0;        // wall time inside vkQueueSubmit
    double cpuWaitNs = 0;           // wall time inside vkWaitForFences
    double frameTotalNs = 0;        // record start -> fence signalled
};
```

The result of one simulation step. `rawTicks[6]` is the raw GPU clock
values; the `...Ns` fields are the derived per-stage durations in
nanoseconds. `gpuValid` is false if timestamps couldn't be read (e.g.
hardware without support). `uint64_t rawTicks[6] = {};` — a C array of six
64-bit ints, `= {}` zeroes them.

The five stage fields are consecutive differences of the six ticks;
`gpuTotalNs` is the whole span `ticks[5] - ticks[0]`. By construction the
five stages sum exactly to `gpuTotalNs` — that's the "100% accounted"
reconciliation the report prints.

### The `Simulation` class (lines 58-122)

Public interface:

| Method | Purpose |
|---|---|
| `init(ctx, params, initial)` | create everything, upload the initial particles |
| `destroy()` | release everything |
| `runForceCalcOnce()` | one force dispatch + readback, no integration (used by the Phase 2 test) |
| `runBatch(steps)` | advance `steps` steps in one command buffer / one submit; the core |
| `stepOnce()` | inline: `runBatch(1)` |
| `lastBatchTimings()` | the `std::vector<StepTimings>` from the last `runBatch` |
| `lastTimings()` | just the last element of that |
| `timestampsSupported()` | did the queue family support timestamps? |
| `readAccelerations()` / `readParticles()` | copy the current GPU results out of the mapped readback buffer |

Private: the helper methods (`createBuffers`, `createPipelines`,
`createQueryPool`, `createLoopObjects`, `beginCommands`, `submitAndWait`)
and ~20 member variables — the Vulkan handles, sizes, the reusable
command buffer/fence, the persistent map pointer, and
`lastBatch_` (initialised to a vector holding one default `StepTimings` so
`lastTimings()` is safe to call before any run).

At the bottom of the header, two free functions:

- `cpuForceCalc(particles, params)` — a plain-C++ reimplementation of the
  force shader, for the correctness test.
- `totalMomentum(particles)` — `Σ mᵢvᵢ`, for the conservation test.

---

## 2. `simulation.cpp` file-level bits (lines 1-38)

```cpp
using Clock = std::chrono::steady_clock;
static double nanosBetween(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::nano>(b - a).count();
}
```

The CPU stopwatch helper (Part 01 §15). `static` = private to this file.

```cpp
#ifndef SHADER_DIR
#define SHADER_DIR "shaders"
#endif
```

Fallback if the build didn't inject `SHADER_DIR` (Part 03).

```cpp
namespace {
void check(VkResult r, const char* what) { ... throw std::runtime_error(...); }

struct PushConstants {
    uint32_t particleCount;
    float gravity;
    float softening;
    float dt;
};

VkDeviceSize alignUp(VkDeviceSize v, VkDeviceSize a) { return (v + a - 1) / a * a; }
}  // namespace
```

- `check` — file-local copy of the error-check wrapper.
- `PushConstants` — the **one combined 16-byte block** used by both shaders.
  `force_calc.comp` declares only the first 3 fields; `integrate.comp`
  declares all 4. A shader may declare a prefix of the range.
- `alignUp(v, a)` — round `v` up to the next multiple of `a`. `(v + a - 1) /
  a` is integer "ceil(v/a)", then `* a`. Used to place the accel region at a
  256-byte-aligned offset inside the readback buffer (Vulkan requires
  aligned copy offsets).

---

## 3. `init` (lines 40-58)

```cpp
void Simulation::init(VulkanContext& ctx, const SimParams& params,
                      const std::vector<Particle>& initial) {
    ctx_ = &ctx;
    params_ = params;
    if (initial.size() != params.particleCount)
        throw std::runtime_error("initial particle count mismatch");

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = ctx.computeQueueFamily;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    check(vkCreateCommandPool(ctx.device, &poolInfo, nullptr, &commandPool_),
          "vkCreateCommandPool");

    createBuffers(initial);
    createPipelines();
    createQueryPool();
    createLoopObjects();
}
```

- Store a pointer to the context (`ctx_ = &ctx`) and a copy of the params.
- Sanity check: the caller-supplied particle vector must match
  `particleCount`.
- Create the **command pool** — the allocator that command buffers come
  from. `queueFamilyIndex` ties it to the compute family (command buffers
  from this pool can only be submitted to that family). The flag
  `RESET_COMMAND_BUFFER_BIT` lets us call `vkResetCommandBuffer` on an
  individual buffer to re-record it (needed because `runBatch` reuses one
  command buffer every step).
- Then the four setup helpers in order.

---

## 4. `createQueryPool` (lines 81-99)

```cpp
uint32_t qCount = 0;
vkGetPhysicalDeviceQueueFamilyProperties(ctx.physicalDevice, &qCount, nullptr);
std::vector<VkQueueFamilyProperties> qProps(qCount);
vkGetPhysicalDeviceQueueFamilyProperties(ctx.physicalDevice, &qCount, qProps.data());
timestampsSupported_ = qProps[ctx.computeQueueFamily].timestampValidBits != 0;
timestampPeriod_ = ctx.props.limits.timestampPeriod;

if (!timestampsSupported_) return;

VkQueryPoolCreateInfo info{};
info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
info.queryType = VK_QUERY_TYPE_TIMESTAMP;
info.queryCount = kTimestampsPerStep * kMaxBatch;   // 6 * 64 = 384
check(vkCreateQueryPool(ctx.device, &info, nullptr, &queryPool_), "vkCreateQueryPool");
```

- `timestampValidBits != 0` on the compute queue family → timestamps
  usable. If 0, set the flag false and **don't create the pool**; `runBatch`
  will skip all timestamp work and produce `gpuValid = false`. (On real
  hardware this basically never happens, but it keeps the code honest.)
- `timestampPeriod` (from `props.limits`, cached in Part 05) is nanoseconds
  per GPU clock tick. On MoltenVK it's `1.0`.
- The pool holds 384 timestamp slots — enough for a full 64-step batch.

---

## 5. `createBuffers` (lines 101-156)

### The `makeBuffer` lambda (lines 106-123)

```cpp
auto makeBuffer = [&](VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags memFlags, VkBuffer& buf,
                      VkDeviceMemory& mem) {
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = size;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    check(vkCreateBuffer(ctx.device, &info, nullptr, &buf), "vkCreateBuffer");

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(ctx.device, buf, &req);

    VkMemoryAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = ctx.findMemoryType(req.memoryTypeBits, memFlags);
    check(vkAllocateMemory(ctx.device, &alloc, nullptr, &mem), "vkAllocateMemory");
    check(vkBindBufferMemory(ctx.device, buf, mem, 0), "vkBindBufferMemory");
};
```

The three-step buffer creation from Part 02 §7, factored into a lambda:

1. `vkCreateBuffer` — describe the buffer (`size`, `usage` = what it's
   allowed to be used for, `SHARING_MODE_EXCLUSIVE` = only one queue family
   touches it).
2. `vkGetBufferMemoryRequirements` — ask Vulkan how much memory it really
   needs (`req.size`, may exceed `size` due to alignment) and which memory
   types are acceptable (`req.memoryTypeBits`).
3. `findMemoryType(req.memoryTypeBits, memFlags)` — pick a memory type that
   is acceptable AND has the properties we asked for (`memFlags`).
4. `vkAllocateMemory` + `vkBindBufferMemory` — allocate and attach.

`VkBuffer& buf, VkDeviceMemory& mem` are output references — the lambda
writes the created handles back into the caller's members.

### The four buffers (lines 125-155)

```cpp
particleBytes_ = sizeof(Particle) * params_.particleCount;    // 32 * N
accelBytes_    = sizeof(float) * 4 * params_.particleCount;   // 16 * N  (vec4 per particle)
```

```cpp
makeBuffer(particleBytes_,
           STORAGE_BUFFER_BIT | TRANSFER_DST_BIT | TRANSFER_SRC_BIT,
           DEVICE_LOCAL_BIT, particleBuffer_, particleMemory_);
```

**particleBuffer** — lives in fast GPU memory (`DEVICE_LOCAL`). Usage:
`STORAGE` (shaders read/write it), `TRANSFER_DST` (staging copies *into*
it at startup), `TRANSFER_SRC` (readback copies *out of* it each step).

```cpp
makeBuffer(accelBytes_,
           STORAGE_BUFFER_BIT | TRANSFER_SRC_BIT,
           DEVICE_LOCAL_BIT, accelBuffer_, accelMemory_);
```

**accelBuffer** — `DEVICE_LOCAL`. `STORAGE` (force writes it, integrate
reads it), `TRANSFER_SRC` (the Phase 2 test copies it out).

```cpp
accelReadbackOffset_ = alignUp(particleBytes_, 256);
makeBuffer(accelReadbackOffset_ + accelBytes_, TRANSFER_DST_BIT,
           HOST_VISIBLE_BIT | HOST_COHERENT_BIT, readbackBuffer_, readbackMemory_);
```

**readbackBuffer** — one `HOST_VISIBLE | HOST_COHERENT` buffer big enough
to hold **both** a full particle snapshot (at offset 0) **and** a full accel
snapshot (at offset `accelReadbackOffset_`, rounded up to 256 for
alignment). Only `TRANSFER_DST` (things get copied into it; the CPU then
reads it through a mapping). Using one buffer with two regions avoids a
second allocation.

```cpp
VkBuffer staging = VK_NULL_HANDLE;
VkDeviceMemory stagingMem = VK_NULL_HANDLE;
makeBuffer(particleBytes_, TRANSFER_SRC_BIT,
           HOST_VISIBLE_BIT | HOST_COHERENT_BIT, staging, stagingMem);
void* mapped = nullptr;
check(vkMapMemory(ctx.device, stagingMem, 0, particleBytes_, 0, &mapped), "vkMapMemory");
std::memcpy(mapped, initial.data(), static_cast<size_t>(particleBytes_));
vkUnmapMemory(ctx.device, stagingMem);

VkCommandBuffer cmd = beginCommands();
VkBufferCopy region{0, 0, particleBytes_};        // srcOffset, dstOffset, size
vkCmdCopyBuffer(cmd, staging, particleBuffer_, 1, &region);
submitAndWait(cmd);

vkDestroyBuffer(ctx.device, staging, nullptr);
vkFreeMemory(ctx.device, stagingMem, nullptr);
```

**The staging upload.** `particleBuffer_` is `DEVICE_LOCAL` so the CPU can't
write to it directly. So:

1. Make a temporary `HOST_VISIBLE` **staging** buffer the same size.
2. `vkMapMemory` it, `memcpy` the initial particles in, `vkUnmapMemory`.
3. Record a one-shot command buffer that does
   `vkCmdCopyBuffer(staging → particleBuffer_)`.
4. `submitAndWait` — submit it and block until the GPU finishes the copy.
5. Destroy the staging buffer; it's done its job.

`VkBufferCopy region{0, 0, particleBytes_}` uses **aggregate
initialisation** — the three fields (`srcOffset`, `dstOffset`, `size`) in
order.

---

## 6. `createPipelines` (lines 158-235)

This builds the descriptor plumbing and the two compute pipelines.

### Descriptor set layout (lines 161-173)

```cpp
std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
for (uint32_t i = 0; i < bindings.size(); ++i) {
    bindings[i].binding = i;                               // slot 0, slot 1
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
}
VkDescriptorSetLayoutCreateInfo layoutInfo{};
layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
layoutInfo.bindingCount = 2;
layoutInfo.pBindings = bindings.data();
check(vkCreateDescriptorSetLayout(...&setLayout_), "vkCreateDescriptorSetLayout");
```

The *shape* of a descriptor set: "two storage buffers, slots 0 and 1, used
by the compute stage." Both shaders use this same shape.

### Pipeline layout + push-constant range (lines 175-183)

```cpp
VkPushConstantRange pcRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants)};
VkPipelineLayoutCreateInfo plInfo{};
plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
plInfo.setLayoutCount = 1;
plInfo.pSetLayouts = &setLayout_;
plInfo.pushConstantRangeCount = 1;
plInfo.pPushConstantRanges = &pcRange;
check(vkCreatePipelineLayout(...&pipelineLayout_), "vkCreatePipelineLayout");
```

`VkPushConstantRange{stageFlags, offset, size}` = "the compute stage can see
16 bytes of push constants starting at offset 0." The **pipeline layout**
bundles {the descriptor set layout, the push-constant range} — it's the
"interface" both pipelines share.

### The `makePipeline` lambda (lines 185-200)

```cpp
auto makePipeline = [&](const char* file, VkPipeline& out) {
    VkShaderModule mod = loadShaderModule(ctx.device, std::string(SHADER_DIR) + "/" + file);
    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = mod;
    stage.pName = "main";                                    // entry point
    VkComputePipelineCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    info.stage = stage;
    info.layout = pipelineLayout_;
    check(vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &info, nullptr, &out),
          "vkCreateComputePipelines");
    vkDestroyShaderModule(ctx.device, mod, nullptr);         // pipeline keeps what it needs
};
makePipeline("force_calc.comp.spv", forceCalcPipeline_);
makePipeline("integrate.comp.spv", integratePipeline_);
```

For each shader: load the `.spv` into a module, wrap it in a
"shader stage" descriptor (`stage` = compute, `pName` = `"main"` the entry
function), create the compute pipeline against the shared layout, then
destroy the module (the pipeline has been fully built). `VK_NULL_HANDLE` in
the middle is the optional "pipeline cache," unused here.

### Descriptor pool + set + writes (lines 204-234)

```cpp
VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};   // 2 storage buffers total
VkDescriptorPoolCreateInfo poolInfo{};
poolInfo.sType = ...DESCRIPTOR_POOL_CREATE_INFO;
poolInfo.maxSets = 1;
poolInfo.poolSizeCount = 1;
poolInfo.pPoolSizes = &poolSize;
check(vkCreateDescriptorPool(...&descriptorPool_), "vkCreateDescriptorPool");
```

The pool is sized for exactly what we need: 1 set, containing 2 storage
buffers.

```cpp
VkDescriptorSetAllocateInfo dsAlloc{};
dsAlloc.sType = ...DESCRIPTOR_SET_ALLOCATE_INFO;
dsAlloc.descriptorPool = descriptorPool_;
dsAlloc.descriptorSetCount = 1;
dsAlloc.pSetLayouts = &setLayout_;
check(vkAllocateDescriptorSets(...&descriptorSet_), "vkAllocateDescriptorSets");
```

Allocate one descriptor set of the shape `setLayout_` from the pool.

```cpp
VkDescriptorBufferInfo particleInfo{particleBuffer_, 0, particleBytes_};  // buffer, offset, range
VkDescriptorBufferInfo accelInfo{accelBuffer_, 0, accelBytes_};
std::array<VkWriteDescriptorSet, 2> writes{};
writes[0].sType = ...WRITE_DESCRIPTOR_SET;
writes[0].dstSet = descriptorSet_;
writes[0].dstBinding = 0;
writes[0].descriptorCount = 1;
writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
writes[0].pBufferInfo = &particleInfo;
writes[1] = writes[0];              // copy, then change the two differing fields
writes[1].dstBinding = 1;
writes[1].pBufferInfo = &accelInfo;
vkUpdateDescriptorSets(ctx.device, 2, writes.data(), 0, nullptr);
```

This is the actual wiring: **slot 0 ← particleBuffer, slot 1 ← accelBuffer**.
`writes[1] = writes[0];` copies the whole struct then overrides just the two
fields that differ — a compact way to fill in two near-identical structs.
`vkUpdateDescriptorSets` commits it. After this, any dispatch that binds
`descriptorSet_` sees the right buffers in the right slots.

Both pipelines share this one descriptor set because both use the same two
buffers in the same slots.

---

## 7. `createLoopObjects` (lines 60-79)

```cpp
VkCommandBufferAllocateInfo cbAlloc{};
cbAlloc.sType = ...COMMAND_BUFFER_ALLOCATE_INFO;
cbAlloc.commandPool = commandPool_;
cbAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
cbAlloc.commandBufferCount = 1;
check(vkAllocateCommandBuffers(ctx.device, &cbAlloc, &loopCmd_), "vkAllocateCommandBuffers");

VkFenceCreateInfo fenceInfo{};
fenceInfo.sType = ...FENCE_CREATE_INFO;
check(vkCreateFence(ctx.device, &fenceInfo, nullptr, &loopFence_), "vkCreateFence");

check(vkMapMemory(ctx.device, readbackMemory_, 0, VK_WHOLE_SIZE, 0, &readbackMapped_),
      "vkMapMemory");
```

Three things allocated **once** and reused every step:

- `loopCmd_` — one **primary** command buffer. `runBatch` calls
  `vkResetCommandBuffer` + re-record instead of allocating a fresh one each
  time.
- `loopFence_` — one fence, `vkResetFences` + reuse each step.
- `readbackMapped_` — a **persistent mapping** of the whole readback buffer
  (`VK_WHOLE_SIZE`). Because the memory is `HOST_COHERENT`, once mapped you
  can just read from `readbackMapped_` any time; no per-step
  `vkMapMemory`/`vkUnmapMemory`.

These three "hoist out of the loop" changes were made during the Phase 8
bottleneck work (Part 12). On MoltenVK their effect was within noise, but
they're correct and matter on a native driver, so they stayed.

---

## 8. `beginCommands` and `submitAndWait` (lines 237-268)

These are the **one-shot** helpers used only for the startup staging upload
and the Phase-2-test `runForceCalcOnce` — *not* the hot loop.

```cpp
VkCommandBuffer Simulation::beginCommands() {
    // allocate ONE fresh command buffer from commandPool_
    // vkBeginCommandBuffer with ONE_TIME_SUBMIT flag
    return cmd;
}

void Simulation::submitAndWait(VkCommandBuffer cmd) {
    check(vkEndCommandBuffer(cmd), ...);
    // create a fresh fence
    // vkQueueSubmit(cmd, fence)
    // vkWaitForFences(fence)
    // vkDestroyFence + vkFreeCommandBuffers   <-- allocate/free each call
}
```

`VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT` tells the driver "I'll submit
this exactly once, feel free to optimise for that." `submitAndWait`
allocates and frees a fence and command buffer per call — fine for
init-time work that happens a handful of times, deliberately *not* how the
per-step loop works.

---

## Explain it in one sentence

> `Simulation::init` creates a command pool then, via helpers, creates
> DEVICE_LOCAL particle and accel buffers plus a HOST_VISIBLE readback
> buffer (uploading the initial particles through a temporary staging
> buffer), builds a two-storage-buffer descriptor set and two compute
> pipelines sharing one pipeline layout, sizes a 384-slot timestamp query
> pool, and allocates one reusable command buffer + fence + a persistent
> readback mapping for the per-step loop.

## Questions you should be able to answer

1. Why does the initial particle upload need a staging buffer at all?
2. Why is the readback buffer one buffer with two regions instead of two
   buffers? What is `accelReadbackOffset_` / `alignUp` for?
3. What does a descriptor set layout describe vs a descriptor set?
4. Why do both pipelines share one descriptor set and one pipeline layout?
5. What does `vkUpdateDescriptorSets` actually connect, and what does the
   `writes[1] = writes[0];` line save you?
6. Why are `loopCmd_`, `loopFence_`, and `readbackMapped_` created once in
   `createLoopObjects` instead of per step?
7. What's the difference between `beginCommands`/`submitAndWait` and how
   `runBatch` handles command buffers?
8. What is the `PushConstants` struct, and why can `force_calc.comp` declare
   only 3 of its 4 fields?
