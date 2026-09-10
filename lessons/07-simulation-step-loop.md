# Part 07 — `simulation.cpp`, part 2: the step loop (`runBatch`)

This is the heart of the project. If you can walk an interviewer through
`runBatch` line by line, you understand the codebase.

Covered here: `runForceCalcOnce`, **`runBatch`** (lines 295-424),
`readAccelerations` / `readParticles`, `destroy`, `cpuForceCalc`,
`totalMomentum`.

---

## 1. `runForceCalcOnce` (lines 270-293) — the warm-up act

Used only by the Phase 2 correctness test. It runs **just** the force
dispatch and copies the acceleration buffer out — no integration, no
timestamps.

```cpp
void Simulation::runForceCalcOnce() {
    VkCommandBuffer cmd = beginCommands();                       // fresh one-shot cmd buffer
    PushConstants pc{params_.particleCount, params_.gravity, params_.softening, params_.dt};

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, forceCalcPipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_,
                            0, 1, &descriptorSet_, 0, nullptr);
    vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    uint32_t groups = (params_.particleCount + 63) / 64;
    vkCmdDispatch(cmd, groups, 1, 1);

    VkMemoryBarrier toTransfer{};
    toTransfer.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    toTransfer.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &toTransfer, 0, nullptr, 0, nullptr);

    VkBufferCopy region{0, accelReadbackOffset_, accelBytes_};
    vkCmdCopyBuffer(cmd, accelBuffer_, readbackBuffer_, 1, &region);

    submitAndWait(cmd);
}
```

Read it as a sentence: bind the force pipeline and the buffers, push the
params, dispatch `ceil(N/64)` workgroups, put a barrier so the shader's
writes to `accelBuffer_` are visible to the following copy, copy
`accelBuffer_` into the accel region of the readback buffer, submit and
wait.

The recording calls:

- `vkCmdBindPipeline(cmd, COMPUTE, pipeline)` — "use this pipeline for the
  next dispatches."
- `vkCmdBindDescriptorSets(cmd, COMPUTE, layout, firstSet=0, count=1,
  &set, dynamicOffsetCount=0, nullptr)` — "slot 0/1 come from this set."
- `vkCmdPushConstants(cmd, layout, stage, offset=0, size, &pc)` — copy the
  16-byte struct into the command stream.
- `vkCmdDispatch(cmd, gx, gy, gz)` — run `gx*gy*gz` workgroups.
- `vkCmdPipelineBarrier(cmd, srcStage, dstStage, depFlags=0,
  memBarrierCount, &memBarrier, bufCount=0, null, imgCount=0, null)` — the
  synchronisation (Part 02 §10, Part 11).
- `vkCmdCopyBuffer(cmd, src, dst, regionCount, &region)` — GPU memcpy.

`readAccelerations()` then reads the result out of the mapped readback
buffer.

---

## 2. `runBatch(uint32_t steps)` — the full thing

The signature: advance `steps` simulation steps, all recorded into **one**
command buffer and issued with **one** `vkQueueSubmit` and **one**
`vkWaitForFences`. Per-step timings are written into `lastBatch_`.
`stepOnce()` is just `runBatch(1)`.

### 2.1 Clamp and start the CPU clock (lines 295-301)

```cpp
void Simulation::runBatch(uint32_t steps) {
    if (steps == 0) steps = 1;
    if (steps > kMaxBatch) steps = kMaxBatch;          // never exceed the query pool

    auto& ctx = *ctx_;
    const bool ts = timestampsSupported_;
    const Clock::time_point tRecordStart = Clock::now();
```

`ts` is a local shorthand for "should I do timestamp work?" `tRecordStart`
marks the beginning of the whole operation (used for `frameTotalNs`).

### 2.2 Re-record the reusable command buffer (lines 303-309)

```cpp
    check(vkResetCommandBuffer(loopCmd_, 0), "vkResetCommandBuffer");
    VkCommandBuffer cmd = loopCmd_;
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(cmd, &begin), "vkBeginCommandBuffer");
```

`vkResetCommandBuffer(loopCmd_, 0)` wipes last step's recording (allowed
because the pool was created with `RESET_COMMAND_BUFFER_BIT`).
`vkBeginCommandBuffer` opens it for recording. `ONE_TIME_SUBMIT` = "this
recording will be submitted once then reset again."

`cmd` is just a short local alias for `loopCmd_`.

### 2.3 Constants and query-pool reset (lines 311-318)

```cpp
    PushConstants pc{params_.particleCount, params_.gravity, params_.softening, params_.dt};
    uint32_t groups = (params_.particleCount + 63) / 64;

    if (ts) vkCmdResetQueryPool(cmd, queryPool_, 0, kTimestampsPerStep * steps);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_,
                            0, 1, &descriptorSet_, 0, nullptr);
    vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
```

- `groups` = `ceil(N / 64)` workgroups per dispatch.
- `vkCmdResetQueryPool(cmd, pool, firstQuery=0, count=6*steps)` — a
  **recorded** command that clears the timestamp slots we're about to
  write. Must happen before any `vkCmdWriteTimestamp` to the same slots or
  results are undefined.
- Bind the descriptor set and push the constants **once** — they don't
  change between steps within a batch, so no need to re-bind them each
  iteration.

### 2.4 Two recording helper lambdas (lines 320-330)

```cpp
    auto memBarrier = [&](VkAccessFlags src, VkAccessFlags dst,
                          VkPipelineStageFlags srcS, VkPipelineStageFlags dstS) {
        VkMemoryBarrier b{};
        b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        b.srcAccessMask = src;
        b.dstAccessMask = dst;
        vkCmdPipelineBarrier(cmd, srcS, dstS, 0, 1, &b, 0, nullptr, 0, nullptr);
    };
    auto stamp = [&](uint32_t q, VkPipelineStageFlagBits stage) {
        if (ts) vkCmdWriteTimestamp(cmd, stage, queryPool_, q);
    };
```

- `memBarrier(srcAccess, dstAccess, srcStage, dstStage)` — record one
  global memory barrier. Used three times per step.
- `stamp(queryIndex, stage)` — record `vkCmdWriteTimestamp` into slot
  `queryIndex` (or do nothing if timestamps unsupported). "Write a timestamp
  when the GPU reaches `stage`." Which stage you pass matters (Part 11).

### 2.5 The per-step recording loop (lines 332-369)

```cpp
    for (uint32_t s = 0; s < steps; ++s) {
        const uint32_t base = kTimestampsPerStep * s;   // this step's first query slot
        const bool last = (s + 1 == steps);
```

`base` = `6 * s` — step `s`'s six timestamps live at slots `base+0 …
base+5`. `last` = is this the final step in the batch?

```cpp
        stamp(base + 0, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, forceCalcPipeline_);
        vkCmdDispatch(cmd, groups, 1, 1);
        stamp(base + 1, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
```

- **Timestamp 0**, at `TOP_OF_PIPE` — "as early as possible," i.e. right
  before this step's GPU work. This is the step's start marker.
- Bind the **force** pipeline, dispatch it.
- **Timestamp 1**, at `COMPUTE_SHADER` — "once compute-shader work up to
  here is done." So `ticks[1] - ticks[0]` = **force dispatch time**.

```cpp
        // accel: shader write -> shader read, before integrate consumes it.
        memBarrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        stamp(base + 2, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
```

- **Barrier 1**: "finish compute-shader writes (`accelBuffer_`) before
  compute-shader reads." This is the force→integrate hazard.
- **Timestamp 2**, at `COMPUTE_SHADER`. `ticks[2] - ticks[1]` = **barrier 1
  stall time** (how long the GPU waited for the write to become visible).

```cpp
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, integratePipeline_);
        vkCmdDispatch(cmd, groups, 1, 1);
        stamp(base + 3, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
```

- Bind **integrate**, dispatch it.
- **Timestamp 3**. `ticks[3] - ticks[2]` = **integrate dispatch time**.

```cpp
        if (last) {
            memBarrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT);
            stamp(base + 4, VK_PIPELINE_STAGE_TRANSFER_BIT);
            VkBufferCopy region{0, 0, particleBytes_};
            vkCmdCopyBuffer(cmd, particleBuffer_, readbackBuffer_, 1, &region);
            stamp(base + 5, VK_PIPELINE_STAGE_TRANSFER_BIT);
        } else {
            memBarrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            stamp(base + 4, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            stamp(base + 5, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);   // no copy this step
        }
    }
```

The last step of the batch differs from the others:

- **Last step:**
  - **Barrier 2** = "finish compute-shader writes (`particleBuffer_`) before
    the **transfer** stage reads them." (A `vkCmdCopyBuffer` runs in the
    `TRANSFER` stage, not compute.)
  - **Timestamp 4** at `TRANSFER`. `ticks[4] - ticks[3]` = **barrier 2 stall
    time**.
  - `vkCmdCopyBuffer(particleBuffer_ → readbackBuffer_ offset 0)` — copy the
    fresh particle snapshot to host-visible memory.
  - **Timestamp 5** at `TRANSFER`. `ticks[5] - ticks[4]` = **readback copy
    time**.

- **Non-last step:**
  - **Barrier 2** is instead "finish compute-shader writes before
    compute-shader reads" — because the *next* step's `force_calc` needs to
    read the just-updated positions. No host round-trip; the dependency is
    expressed as an in-queue barrier.
  - Timestamps 4 and 5 are both written at `COMPUTE_SHADER`, right after the
    barrier, so `ticks[4] - ticks[3]` = that barrier's stall and
    `ticks[5] - ticks[4]` = ~0 (no copy this step; `copyNs` ≈ 0).

**This is the batching mechanism.** With `steps = 1`, only the "last" branch
runs → identical to the original one-step behaviour. With `steps = 32`, 31
steps chain through in-queue barriers and only the 32nd does a readback and
is followed by a single submit+wait.

### 2.6 End recording, submit, wait — with CPU timing (lines 371-384)

```cpp
    check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
    check(vkResetFences(ctx.device, 1, &loopFence_), "vkResetFences");

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;

    const Clock::time_point tBeforeSubmit = Clock::now();
    check(vkQueueSubmit(ctx.computeQueue, 1, &submit, loopFence_), "vkQueueSubmit");
    const Clock::time_point tAfterSubmit = Clock::now();
    check(vkWaitForFences(ctx.device, 1, &loopFence_, VK_TRUE, UINT64_MAX), "vkWaitForFences");
    const Clock::time_point tAfterWait = Clock::now();
```

- `vkEndCommandBuffer` — finish recording.
- `vkResetFences` — put the fence back to "unsignalled" so this submission
  can signal it.
- `vkQueueSubmit(queue, submitCount=1, &submitInfo, fence)` — hand the
  command buffer to the GPU; the GPU will signal `loopFence_` when done.
- `vkWaitForFences(device, count=1, &fence, waitAll=VK_TRUE,
  timeout=UINT64_MAX)` — block the CPU until the fence is signalled (no
  timeout).

Three clock reads bracket the two blocking calls:

- `tAfterSubmit - tBeforeSubmit` → time spent *inside* `vkQueueSubmit`.
- `tAfterWait - tAfterSubmit` → time spent *inside* `vkWaitForFences` (≈ how
  long the GPU actually took, plus signal latency).
- `tAfterWait - tRecordStart` → total wall-clock for the whole operation
  (record + submit + wait).

### 2.7 Amortise the CPU costs across the batch (lines 386-390)

```cpp
    const double submitPer = nanosBetween(tBeforeSubmit, tAfterSubmit) / steps;
    const double waitPer   = nanosBetween(tAfterSubmit, tAfterWait) / steps;
    const double framePer  = nanosBetween(tRecordStart, tAfterWait) / steps;
```

There's **one** submit and **one** wait for the whole batch, but the report
aggregates *per step*. So each step is charged an **equal share**: total
submit time / `steps`, etc. This is what makes `batch=1` and `batch=32`
runs directly comparable — the "per step" numbers mean the same thing. It's
also exactly why batching *helps*: a fixed ~0.5 ms round-trip divided by 32
is ~16 µs per step instead of ~500 µs (Part 12).

### 2.8 Read back the GPU timestamps (lines 392-400)

```cpp
    std::vector<uint64_t> ticks(kTimestampsPerStep * steps);
    bool gpuOk = false;
    if (ts) {
        VkResult r = vkGetQueryPoolResults(
            ctx.device, queryPool_, 0, kTimestampsPerStep * steps,
            ticks.size() * sizeof(uint64_t), ticks.data(), sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        gpuOk = (r == VK_SUCCESS);
    }
```

`vkGetQueryPoolResults(device, pool, firstQuery=0, queryCount=6*steps,
dataSizeBytes, dataPtr, strideBytes=8, flags)`:

- writes each result as a `uint64_t` (`VK_QUERY_RESULT_64_BIT`),
- `VK_QUERY_RESULT_WAIT_BIT` — block until all those timestamps are
  available (they will be, since the fence already signalled, but this makes
  it explicit and safe),
- stride 8 bytes = one `uint64_t` per slot, tightly packed into `ticks`.

`gpuOk` is true only if it succeeded.

### 2.9 Fill in `lastBatch_` (lines 402-423)

```cpp
    lastBatch_.assign(steps, StepTimings{});
    for (uint32_t s = 0; s < steps; ++s) {
        StepTimings& out = lastBatch_[s];
        out.timestampPeriodNs = timestampPeriod_;
        out.submitCallNs = submitPer;
        out.cpuWaitNs = waitPer;
        out.frameTotalNs = framePer;
        if (gpuOk) {
            const uint64_t* t = &ticks[kTimestampsPerStep * s];   // this step's 6 ticks
            out.gpuValid = true;
            for (uint32_t i = 0; i < kTimestampsPerStep; ++i) out.rawTicks[i] = t[i];
            auto span = [&](int a, int b) {
                return static_cast<double>(t[b] - t[a]) * timestampPeriod_;
            };
            out.forceDispatchNs     = span(0, 1);
            out.barrier1Ns          = span(1, 2);
            out.integrateDispatchNs = span(2, 3);
            out.barrier2Ns          = span(3, 4);
            out.copyNs              = span(4, 5);
            out.gpuTotalNs          = span(0, 5);
        }
    }
}
```

- `lastBatch_.assign(steps, StepTimings{})` — resize `lastBatch_` to hold
  `steps` fresh default `StepTimings`.
- For each step, copy in the amortised CPU numbers, then if the GPU
  timestamps came back:
  - `t` points at this step's six ticks (`&ticks[6*s]`).
  - `span(a, b)` = `(ticks[b] - ticks[a]) * timestampPeriod` → nanoseconds
    between two markers.
  - The five stage durations are consecutive spans; `gpuTotalNs` is the
    whole `0→5` span. They sum to `gpuTotalNs` exactly (each internal tick
    is both the end of one span and the start of the next), which is the
    "100% accounted" line in the report.

After `runBatch` returns, `collectSteps` (in `main.cpp`) pushes every
`StepTimings` in `lastBatch_` into the `Report`.

---

## 3. `readAccelerations` / `readParticles` (lines 426-439)

```cpp
std::vector<Vec3> Simulation::readAccelerations() {
    const float* f = reinterpret_cast<const float*>(
        static_cast<const char*>(readbackMapped_) + accelReadbackOffset_);
    std::vector<Vec3> out(params_.particleCount);
    for (uint32_t i = 0; i < params_.particleCount; ++i)
        out[i] = {f[i * 4 + 0], f[i * 4 + 1], f[i * 4 + 2]};
    return out;
}
```

`readbackMapped_` is a `void*` to the whole readback buffer. The accel data
starts `accelReadbackOffset_` bytes in.

- `static_cast<const char*>(readbackMapped_) + accelReadbackOffset_` — do
  byte-precise pointer arithmetic (`char` is 1 byte) to reach the accel
  region.
- `reinterpret_cast<const float*>(...)` — "the bytes there are floats."
- Each particle's accel is a `vec4` = 4 floats; take indices `4i+0/1/2`
  (xyz), skip `4i+3` (the unused w). `out[i] = {a, b, c};` builds a `Vec3`.

No `vkMapMemory`/`vkUnmapMemory` — the mapping is persistent
(`createLoopObjects`) and the memory is `HOST_COHERENT` so GPU writes are
already visible.

```cpp
std::vector<Particle> Simulation::readParticles() {
    std::vector<Particle> out(params_.particleCount);
    std::memcpy(out.data(), readbackMapped_, static_cast<size_t>(particleBytes_));
    return out;
}
```

Particles start at offset 0 and the CPU `Particle` layout matches the GPU's
exactly, so it's a single `memcpy` of `particleBytes_` bytes.

---

## 4. `destroy` (lines 441-464)

```cpp
void Simulation::destroy() {
    if (!ctx_) return;                    // never inited / already destroyed
    VkDevice d = ctx_->device;
    vkDeviceWaitIdle(d);                  // <-- CRUCIAL: wait for all GPU work to finish
    if (readbackMapped_) { vkUnmapMemory(d, readbackMemory_); readbackMapped_ = nullptr; }
    if (loopFence_)  vkDestroyFence(d, loopFence_, nullptr);
    if (queryPool_)  vkDestroyQueryPool(d, queryPool_, nullptr);
    if (forceCalcPipeline_) vkDestroyPipeline(d, forceCalcPipeline_, nullptr);
    if (integratePipeline_) vkDestroyPipeline(d, integratePipeline_, nullptr);
    if (pipelineLayout_)    vkDestroyPipelineLayout(d, pipelineLayout_, nullptr);
    if (descriptorPool_)    vkDestroyDescriptorPool(d, descriptorPool_, nullptr);
    if (setLayout_)         vkDestroyDescriptorSetLayout(d, setLayout_, nullptr);
    if (particleBuffer_)    vkDestroyBuffer(d, particleBuffer_, nullptr);
    if (particleMemory_)    vkFreeMemory(d, particleMemory_, nullptr);
    if (accelBuffer_)       vkDestroyBuffer(d, accelBuffer_, nullptr);
    if (accelMemory_)       vkFreeMemory(d, accelMemory_, nullptr);
    if (readbackBuffer_)    vkDestroyBuffer(d, readbackBuffer_, nullptr);
    if (readbackMemory_)    vkFreeMemory(d, readbackMemory_, nullptr);
    if (commandPool_)       vkDestroyCommandPool(d, commandPool_, nullptr);
    ctx_ = nullptr;
}
```

- `vkDeviceWaitIdle` first — you must not destroy an object the GPU might
  still be using. This blocks until the GPU has nothing left to do.
- Unmap before freeing the memory it maps.
- Everything else destroyed, each guarded by `if (handle)` so a
  half-initialised object tears down cleanly.
- Descriptor sets are freed automatically when the pool is destroyed (no
  explicit `vkFreeDescriptorSets`). The reusable command buffer is freed
  with the command pool.
- `ctx_ = nullptr` makes a second `destroy()` call a safe no-op.

The self-tests create and destroy a `Simulation` per test case, so this runs
dozens of times per program run without leaking.

---

## 5. `cpuForceCalc` (lines 466-489) — the reference implementation

```cpp
std::vector<Vec3> cpuForceCalc(const std::vector<Particle>& particles,
                               const SimParams& params) {
    std::vector<Vec3> accel(particles.size());
    const float soft2 = params.softening * params.softening;
    for (size_t i = 0; i < particles.size(); ++i) {
        float ax = 0, ay = 0, az = 0;
        const float pix = particles[i].px, piy = particles[i].py, piz = particles[i].pz;
        for (size_t j = 0; j < particles.size(); ++j) {
            if (j == i) continue;
            const float dx = particles[j].px - pix;
            const float dy = particles[j].py - piy;
            const float dz = particles[j].pz - piz;
            const float distSqr = dx*dx + dy*dy + dz*dz + soft2;
            const float invDist = 1.0f / std::sqrt(distSqr);
            const float invDist3 = invDist * invDist * invDist;
            const float s = params.gravity * particles[j].mass * invDist3;
            ax += s * dx; ay += s * dy; az += s * dz;
        }
        accel[i] = {ax, ay, az};
    }
    return accel;
}
```

A line-for-line CPU version of `force_calc.comp`, in plain C++ scalar math.
**Same loop order** (`j = 0 … N-1`), same softening, same formula. The Phase
2 test runs this and the GPU shader on the *same* particles and checks they
agree to a tight tolerance (~1e-7 relative error). Slight differences are
expected because the GPU's `inversesqrt` and the CPU's `1.0f / std::sqrt`
aren't bit-identical, and floating-point sums depend on order — hence a
tolerance, not exact equality.

This is the project's answer to "how do you know the GPU math is right?" —
you don't trust it, you cross-check it against an independent
implementation.

---

## 6. `totalMomentum` (lines 491-499)

```cpp
Vec3 totalMomentum(const std::vector<Particle>& particles) {
    double mx = 0, my = 0, mz = 0;
    for (const auto& p : particles) {
        mx += static_cast<double>(p.mass) * p.vx;
        my += static_cast<double>(p.mass) * p.vy;
        mz += static_cast<double>(p.mass) * p.vz;
    }
    return {static_cast<float>(mx), static_cast<float>(my), static_cast<float>(mz)};
}
```

`Σ mᵢ vᵢ`, accumulated in `double` (to minimise summation error over
thousands of particles) then narrowed back to `float` for the `Vec3`. The
Phase 3 test computes this before the run and after every step; the
difference ("drift") should stay tiny because momentum is conserved (Part
04 §2).

---

## 7. Why `runBatch` is the whole project in one function

- It records the pipeline: **dispatch → barrier → dispatch → barrier →
  copy**.
- It instruments every boundary with a timestamp so each stage's GPU time is
  measured.
- It times the CPU side (submit + fence wait) with `std::chrono`.
- It batches steps to amortise the fixed per-submit cost — the knob the
  bottleneck fix turns.
- It produces `StepTimings`, which the `Report` turns into the table.

Every other file is either "get to the point where `runBatch` can run"
(`vk_context`, the setup half of `simulation`) or "do something with what
`runBatch` produced" (`report`, `main`).

---

## Explain it in one sentence

> `runBatch` resets and re-records one reusable command buffer with `steps`
> iterations of (timestamp, force dispatch, timestamp, barrier, timestamp,
> integrate dispatch, timestamp, barrier, timestamp[, copy on the last
> step], timestamp), submits it once, blocks on one fence while timing the
> submit and wait with `std::chrono`, reads the GPU timestamps back, and
> fills `lastBatch_` with one `StepTimings` per step whose five stage
> durations are consecutive timestamp differences and whose CPU numbers are
> the single submit/wait cost divided evenly across the batch.

## Questions you should be able to answer

1. Walk through the six timestamps in one step: which stage flag does each
   use, and what duration does each *pair* give you?
2. How does the last step of a batch differ from the others, and why?
3. With `steps = 1`, which branch of the `if (last)` runs, and why does that
   make `runBatch(1)` behave like the original one-step loop?
4. Why are `submitCallNs` / `cpuWaitNs` / `frameTotalNs` divided by `steps`?
5. Why does `runBatch` reset and reuse `loopCmd_` instead of allocating a
   command buffer each call?
6. Why must `destroy()` call `vkDeviceWaitIdle` before destroying anything?
7. What is `cpuForceCalc` for, and why does the test use a *tolerance*
   instead of checking exact equality?
8. Why is `totalMomentum` accumulated in `double`?
9. Why do the five per-stage durations always sum exactly to `gpuTotalNs`?
