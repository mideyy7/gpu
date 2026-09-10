# Part 02 — The GPU and Vulkan mental model

No project code in this part. This is the conceptual foundation. If you
understand this part, the rest of the codebase is just "typing it out."

---

## 1. CPU vs GPU

A **CPU** has a handful of powerful cores. It's great at doing one
complicated thing after another, with lots of branching.

A **GPU** has thousands of tiny cores. It's great at doing the *same simple
thing* to thousands of pieces of data at once. "For every one of these
100,000 particles, compute its acceleration" is exactly the shape of problem
a GPU eats for breakfast.

The catch: the GPU is a **separate device**. It has its own memory. To use
it you must:

1. copy your data into GPU memory,
2. tell it what program to run and how many times,
3. wait for it to finish,
4. copy the results back.

Steps 1–4 have real overhead, and this project's whole job is to *measure*
that overhead, stage by stage.

---

## 2. "Compute" vs "graphics"

GPUs were built to draw triangles. But the same parallel hardware can run
**general-purpose programs** that just read and write arrays of numbers,
with no triangles involved. That's **compute**.

This project is pure compute: no window, no image on screen ("headless"). It
runs two small programs — one that computes forces, one that moves particles
— over big arrays. The programs are called **compute shaders**.

---

## 3. Driver, API, and the Vulkan loader

- Your GPU's manufacturer ships a **driver**: the software that actually
  talks to the hardware.
- **Vulkan** is a standard **API** (a fixed set of C functions like
  `vkCreateBuffer`) that every conformant driver implements. Write to Vulkan
  once, run on any GPU whose driver supports it.
- The **Vulkan loader** is a thin library (`libvulkan`) that sits between
  your program and the driver. When you call `vkCreateBuffer`, you're really
  calling the loader, which routes the call to the right driver (called an
  **ICD** — Installable Client Driver).

```
your program  ─►  Vulkan loader  ─►  [layers]  ─►  driver (ICD)  ─►  GPU
```

- **Layers** are optional plugins the loader can insert into that chain.
  The most important is the **validation layer**
  (`VK_LAYER_KHRONOS_validation`), which checks every call for correctness
  and screams if you misuse the API. This project runs with it on at all
  times and treats any warning as a bug. Part 10 builds a *custom* layer.

### On this machine (macOS): MoltenVK

macOS has no native Vulkan driver. **MoltenVK** is an ICD that translates
Vulkan calls into Apple's **Metal** API. It works, but that translation adds
overhead, so absolute timing numbers here are "indicative, not native." The
code detects this (`isTranslationLayer`) and prints a warning. On a Linux
machine with a real GPU driver, the same binary runs unchanged.

---

## 4. The Vulkan object hierarchy

You build these in order. Each one is a **handle** — an opaque `uint64_t`
you pass back to Vulkan; you never look inside it.

### `VkInstance`

Your connection to the Vulkan loader. Created with `vkCreateInstance`. You
tell it which **instance-level extensions** and **layers** you want (e.g.
"enable the validation layer," "enable the debug-messenger extension").
One per program.

### `VkPhysicalDevice`

A GPU that Vulkan can see. `vkEnumeratePhysicalDevices` lists them. You query
its **properties** (name, limits, driver version) and its **memory
properties** (what kinds of memory it has) and its **queue families** (see
below), then pick one. You don't create or destroy it — it just *is*.

### Queue families and `VkQueue`

A GPU accepts work through **queues**. Queues come in **families**, and each
family advertises capabilities: graphics, compute, transfer (memory copies).
This project scans for a family with the **compute** bit
(`VK_QUEUE_COMPUTE_BIT`) and remembers its index (`computeQueueFamily`).

### `VkDevice` (the "logical device")

Your actual, usable connection to one physical device. Created with
`vkCreateDevice`, where you say "give me one queue from family N" and "enable
these **device-level extensions**." Almost every other Vulkan function takes
a `VkDevice` as its first argument. When you're done: `vkDestroyDevice`.

After creating it you call `vkGetDeviceQueue` to get the `VkQueue` handle you
submit work to.

`vk_context.cpp` does all of the above, in that order. That's its entire
job.

### The `sType` field

Every Vulkan "create info" struct has a first field called `sType` set to a
constant like `VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO`. Why? Vulkan structs
can be *extended*: a struct can point (via a `pNext` pointer) to another
struct carrying extra options. The driver walks that chain and uses `sType`
to know what each link is. This project doesn't use `pNext` chains, but it
must still set every `sType` or the driver rejects the struct.

---

## 5. Commands: record, submit, wait

You don't tell the GPU to do things one call at a time. You:

1. **Record** a list of commands into a `VkCommandBuffer` (allocated from a
   `VkCommandPool`). Recording is a CPU-side activity — nothing runs yet.
   Commands are functions starting `vkCmd...`: `vkCmdDispatch`,
   `vkCmdPipelineBarrier`, `vkCmdCopyBuffer`, `vkCmdWriteTimestamp`.
2. **Submit** the finished command buffer to a `VkQueue` with
   `vkQueueSubmit`. *Now* the GPU starts working through the list.
3. **Wait** for it to finish, using a **fence** (below).

This "record a batch, submit it, wait" cycle is the beating heart of the
project. `simulation.cpp`'s `runBatch` does exactly this, once per call.

### Why a batch instead of call-by-call?

Because submitting is expensive (the CPU has to hand work across to the
driver and GPU, and later get a signal back). If you submit tiny pieces, you
pay that cost constantly. The **bottleneck fix in part 12** is precisely
"record more steps per submit."

---

## 6. Dispatch: running a compute shader

`vkCmdDispatch(cmd, gx, gy, gz)` says "run the currently-bound compute
shader `gx * gy * gz` **workgroups** times."

A compute shader declares its **workgroup size** in the shader source:
`layout(local_size_x = 64) in;` means "each workgroup is 64 invocations
(threads)."

So if you have 8192 particles and want one shader invocation per particle:

- workgroup size = 64,
- number of workgroups = ceil(8192 / 64) = 128,
- `vkCmdDispatch(cmd, 128, 1, 1)` → 128 × 64 = 8192 invocations.

Inside the shader, `gl_GlobalInvocationID.x` tells each invocation which
global index (0 … 8191) it is, so it knows which particle to work on. If the
particle count isn't a multiple of 64, the last workgroup has a few extra
invocations, so the shader starts with `if (i >= particleCount) return;` to
make them do nothing.

`(count + 63) / 64` is the standard integer trick for "ceil(count / 64)."
You'll see it in `simulation.cpp:278` and `:312`.

---

## 7. Buffers and memory

### Two objects, not one

Getting a chunk of GPU-accessible memory takes **two** steps in Vulkan:

1. `vkCreateBuffer` → a `VkBuffer`: a *description* ("I want 1 MB, usable as
   a storage buffer and a copy source"). No memory yet.
2. `vkAllocateMemory` → a `VkDeviceMemory`: the actual bytes.
3. `vkBindBufferMemory` → glue the two together.

`simulation.cpp`'s `makeBuffer` lambda does all three.

### Memory types

A GPU exposes several **memory types**, each with property flags:

- `DEVICE_LOCAL` — fast memory physically on the GPU. The shaders should
  read/write this.
- `HOST_VISIBLE` — memory the CPU can `map` (get a normal pointer to) and
  read/write directly.
- `HOST_COHERENT` — a HOST_VISIBLE type where CPU and GPU automatically see
  each other's writes (no manual "flush").

On a discrete GPU, `DEVICE_LOCAL` is *not* `HOST_VISIBLE` — the CPU
literally cannot touch it. So to get data in, you:

- put it in a small `HOST_VISIBLE` **staging buffer** (CPU writes it),
- record `vkCmdCopyBuffer(staging → device_local)`,
- submit and wait.

And to get results out, the reverse: `vkCmdCopyBuffer(device_local →
host_visible readback buffer)`, then map the readback buffer and read it.

This project uses:

- `particleBuffer`, `accelBuffer` — `DEVICE_LOCAL` (the shaders use them),
- one `readbackBuffer` — `HOST_VISIBLE | HOST_COHERENT` (the CPU reads
  results from it),
- a temporary `staging` buffer at startup to upload the initial particles.

`findMemoryType` (`vk_context.cpp:167`) picks a memory type index that
satisfies "usable by this buffer" *and* "has all the flags I asked for."

### Mapping

`vkMapMemory(device, memory, ...)` gives you a plain `void*` pointing into a
`HOST_VISIBLE` allocation. Read/write through it like normal memory.
`vkUnmapMemory` releases it. This project maps the readback buffer **once**
at startup and keeps the pointer for the whole run (`readbackMapped_`) —
"persistent mapping," a small optimisation (part 12).

---

## 8. Descriptors: connecting buffers to shader slots

A shader says "I read from buffer slot 0 and write buffer slot 1"
(`layout(binding = 0)`, `layout(binding = 1)`). Something has to tell the
GPU *which actual buffer* is slot 0. That's a **descriptor**.

The machinery (all in `simulation.cpp`'s `createPipelines`):

- **`VkDescriptorSetLayout`** — the *shape*: "slot 0 is a storage buffer for
  the compute stage, slot 1 is a storage buffer for the compute stage."
- **`VkDescriptorPool`** — an allocator you carve descriptor sets from.
- **`VkDescriptorSet`** — an *instance* of that shape with real buffers
  plugged in. `vkUpdateDescriptorSets` writes "slot 0 = particleBuffer,
  slot 1 = accelBuffer" into it.
- At record time, `vkCmdBindDescriptorSets` says "for the next dispatches,
  use this set."

Both shaders in this project use the *same* two buffers in the same slots,
so there's **one** descriptor set layout, **one** descriptor set, shared by
both pipelines.

### Push constants

For a *tiny* amount of data (here: particle count, gravity, softening, dt —
16 bytes total) descriptors are overkill. **Push constants** are a small
block you shove straight into the command buffer with `vkCmdPushConstants`.
Fast, no descriptor set needed. The shader reads them via
`layout(push_constant) uniform PushConstants { ... }`.

The C++ side must define a struct with the **exact same layout**
(`simulation.cpp:29-34`), or the shader reads garbage.

---

## 9. Pipelines

A **pipeline** is a fully "baked" GPU program plus all the fixed state it
needs, ready to run. For compute it's simple:

- **`VkShaderModule`** — the compiled shader code (SPIR-V bytes) wrapped in a
  handle. `loadShaderModule` (`vk_context.cpp:188`) reads a `.spv` file and
  calls `vkCreateShaderModule`.
- **`VkPipelineLayout`** — declares what "interface" the pipeline expects:
  which descriptor set layouts, which push-constant range. Both pipelines
  here share one layout.
- **`VkPipeline`** (compute) — created by `vkCreateComputePipelines` from
  {a shader module, an entry point name `"main"`, a pipeline layout}.

This project builds **two** compute pipelines: `forceCalcPipeline_` from
`force_calc.comp.spv`, `integratePipeline_` from `integrate.comp.spv`. At
record time you `vkCmdBindPipeline` the one you want before each
`vkCmdDispatch`.

### SPIR-V and GLSL

You don't write shaders in machine code. You write **GLSL** (a C-like
language, the `.comp` files), and a compiler (`glslc`, from the Vulkan SDK)
turns it into **SPIR-V** — a portable binary intermediate representation
that all Vulkan drivers understand. `CMakeLists.txt` runs `glslc` on each
`.comp` file at build time (part 03).

---

## 10. Synchronisation: the hazard problem

The GPU runs work **in parallel and out of order** unless you tell it not
to. That creates **hazards**:

- The force shader **writes** `accelBuffer`.
- The integrate shader **reads** `accelBuffer`.

If the GPU is allowed to start the integrate dispatch before the force
dispatch's writes have actually landed in memory, integrate reads stale or
half-written data. This is a **write-after → read** hazard (a "RAW"
dependency).

You fix it with a **pipeline barrier**: `vkCmdPipelineBarrier`. A barrier
says:

> "Before any work in **destination stage** with **destination access** may
> proceed, all prior work in **source stage** with **source access** must
> complete and its memory writes must be visible."

For the force→integrate hazard, the barrier in this code is:

- source stage = `COMPUTE_SHADER`, source access = `SHADER_WRITE`
- dest stage = `COMPUTE_SHADER`, dest access = `SHADER_READ`

Read as: "finish the compute-shader writes before starting compute-shader
reads." (`simulation.cpp:342-344`.)

There's a second barrier before the readback copy: source = compute-shader
write, dest = **transfer** read, because a `vkCmdCopyBuffer` runs in the
`TRANSFER` stage, not the compute stage. (`simulation.cpp:353-355`.)

**Why this matters for the profiler:** the time the GPU spends *stalled* at
a barrier is one of the things being measured. Timestamps are written on
both sides of each barrier so its cost shows up as its own line in the
table. Part 11 goes deep on this.

---

## 11. Fences: telling the CPU the GPU is done

- **`VkFence`** — a signal from GPU back to CPU. You pass a fence to
  `vkQueueSubmit`; the GPU **signals** it when that submission finishes. The
  CPU calls `vkWaitForFences` to **block** until it's signalled. Then
  `vkResetFences` to reuse it.
- **`VkSemaphore`** — GPU-to-GPU synchronisation (e.g. between queues). This
  project has one queue and one submit in flight at a time, so it doesn't
  need semaphores.

The pattern in `runBatch`:

```
record command buffer
vkResetFences(loopFence_)
t0 = now()
vkQueueSubmit(queue, cmd, loopFence_)       // GPU starts
t1 = now()                                  // -> "submit call" CPU time
vkWaitForFences(loopFence_)                 // CPU blocks here until GPU done
t2 = now()                                  // -> "fence wait" CPU time
```

`t2 - t0` is the wall-clock time for the whole step. The GPU-side timestamps
tell you how much of that was actual GPU compute vs. overhead.

---

## 12. Query pools and timestamps

A **`VkQueryPool`** is an array of slots the GPU can write measurements
into. This project creates one of type `VK_QUERY_TYPE_TIMESTAMP`.

- `vkCmdWriteTimestamp(cmd, stage, pool, index)` records a command that,
  when the GPU reaches `stage`, writes the current GPU clock **tick** into
  `pool[index]`.
- `vkCmdResetQueryPool` must clear the slots before they're written each
  time.
- After the fence signals, `vkGetQueryPoolResults` copies the ticks back to
  the CPU.
- Ticks are converted to nanoseconds by multiplying by
  `VkPhysicalDeviceLimits::timestampPeriod` (nanoseconds per tick).
- `timestampValidBits` on the queue family tells you whether timestamps are
  supported at all (0 = not supported).

The project writes **6 timestamps per step** (before force, after force /
before barrier1, after barrier1, after integrate, after barrier2 / before
copy, after copy). Consecutive differences give per-stage GPU durations.
Part 07 and part 11 cover the subtleties of *which stage* you pass and *what
the number actually means*.

---

## 13. Putting it together: one simulation step

Here is the full life of one step, in order:

```
CPU: reset the command buffer
CPU: begin recording
CPU: vkCmdResetQueryPool            (clear timestamp slots)
CPU: vkCmdBindDescriptorSets        (slot 0 = particles, slot 1 = accel)
CPU: vkCmdPushConstants            (particleCount, gravity, softening, dt)
CPU: vkCmdWriteTimestamp #0
CPU: vkCmdBindPipeline(force);  vkCmdDispatch(128,1,1)
CPU: vkCmdWriteTimestamp #1
CPU: vkCmdPipelineBarrier          (compute write -> compute read, on accel)
CPU: vkCmdWriteTimestamp #2
CPU: vkCmdBindPipeline(integrate); vkCmdDispatch(128,1,1)
CPU: vkCmdWriteTimestamp #3
CPU: vkCmdPipelineBarrier          (compute write -> transfer read, on particles)
CPU: vkCmdWriteTimestamp #4
CPU: vkCmdCopyBuffer(particles -> readback)
CPU: vkCmdWriteTimestamp #5
CPU: end recording
CPU: vkResetFences; t0=now
CPU: vkQueueSubmit(..., fence)                 <-- GPU begins here
CPU: t1=now                                     (submit-call duration = t1-t0)
GPU:   runs the recorded list; writes ticks 0..5 as it passes each point
GPU:   signals the fence when done
CPU: vkWaitForFences  (blocks)
CPU: t2=now                                     (fence-wait duration = t2-t1)
CPU: vkGetQueryPoolResults -> ticks[0..5]
CPU: durations: force = (t1-t0)*period, barrier1 = (t2-t1)*period, ...
CPU: read particle results straight from the mapped readback pointer
```

Everything in `simulation.cpp` is an elaboration of this.

---

## Explain it in one sentence

> Vulkan makes you build a small stack of objects (instance → physical
> device → logical device → queue), allocate and bind your own GPU memory,
> compile shaders to SPIR-V and wrap them in pipelines, connect buffers to
> shader slots via descriptors, then record a list of commands into a
> command buffer, submit it to the queue, and wait on a fence — and it does
> nothing implicitly, which is exactly why you can measure every stage.

## Questions you should be able to answer

1. Why does using a GPU involve copying data in and out, and why does that
   matter for a profiler?
2. What is the Vulkan loader, and what is a layer? Name one layer this
   project relies on and one it builds.
3. Put these in creation order and say what each is: `VkQueue`,
   `VkInstance`, `VkDevice`, `VkPhysicalDevice`.
4. What is a workgroup? If you have 5000 items and `local_size_x = 64`, how
   many workgroups do you dispatch, and what does the shader do about the
   leftover invocations?
5. Why does allocating GPU memory take three calls
   (`vkCreateBuffer`/`vkAllocateMemory`/`vkBindBufferMemory`)?
6. What's the difference between `DEVICE_LOCAL` and `HOST_VISIBLE` memory,
   and what is a staging buffer for?
7. What is a descriptor set, and what are push constants used for instead?
8. What is a pipeline barrier, and what specific hazard do the two barriers
   in this project prevent?
9. What is a fence, and what two CPU-side durations does this project
   measure around the submit/wait?
10. How does a GPU timestamp become a nanosecond number?
