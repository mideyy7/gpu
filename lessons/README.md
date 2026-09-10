# Lessons: understanding this codebase end to end

You wrote (well, commissioned) a GPU compute profiler in Vulkan and C++, and
you want to be able to explain **every line** in an interview. These lessons
take you there from a standing start: C++ beginner, never used Vulkan, not
sure what the project even does.

## How to use this

Read the parts **in order**. Each one builds on the last. Every part ends
with:

- **"Explain it in one sentence"** — the interview-ready summary of that part.
- **"Questions you should be able to answer"** — if you can't, re-read.

Keep the actual source open beside the lesson. When a lesson says
`simulation.cpp:295`, open that file to that line and look at it. The goal is
not to memorise the text of the lessons — it's to be able to point at any
line of the real code and say what it does and why.

## The parts

| # | File | What it covers |
|---|------|----------------|
| 00 | `00-orientation.md` | What the project is, the problem it solves, the shape of the whole thing. No code. |
| 01 | `01-cpp-for-this-codebase.md` | Every C++ feature used here, each tied to a real line. |
| 02 | `02-gpu-and-vulkan-model.md` | What a GPU is, what Vulkan is, the objects, the "record → submit → wait" loop. No code. |
| 03 | `03-cmake-and-the-build.md` | `CMakeLists.txt` line by line; how source becomes a binary; SPIR-V. |
| 04 | `04-the-shaders.md` | GLSL; `force_calc.comp` and `integrate.comp` line by line; the physics. |
| 05 | `05-vk_context.md` | `vk_context.h` / `.cpp` line by line — Vulkan start-up. |
| 06 | `06-simulation-setup.md` | `simulation.h` structs; buffers, pipelines, descriptors, query pool. |
| 07 | `07-simulation-step-loop.md` | `runBatch` in depth — the heart of the profiler. |
| 08 | `08-report.md` | `report.h` / `.cpp` — turning raw numbers into a table. |
| 09 | `09-main.md` | `main.cpp` — argument parsing, the self-tests, the run modes. |
| 10 | `10-the-layer.md` | What a Vulkan layer is; `layer/gcp_layer.cpp` in depth. |
| 11 | `11-synchronisation-and-timestamps.md` | Barriers and `vkCmdWriteTimestamp` — the subtle stuff interviewers probe. |
| 12 | `12-the-bottleneck-story.md` | How the profiler found a real bottleneck and how the fix works. |
| 13 | `13-interview-guide.md` | Elevator pitches, likely questions with answers, the weak points and how to own them. |
| — | `glossary.md` | Every term in one place. |

## Suggested pace

- **Day 1:** Parts 00, 01, 02. (Concepts + C++. No rush.)
- **Day 2:** Parts 03, 04, 05.
- **Day 3:** Parts 06, 07. (07 is the big one — take your time.)
- **Day 4:** Parts 08, 09.
- **Day 5:** Parts 10, 11.
- **Day 6:** Parts 12, 13. Then do a mock explanation out loud.

## Building it yourself, in order

If you want to *re-type the whole thing by hand* to really learn it, build it
in these nine steps. **Each step ends with something that compiles and
runs** — never write more than one step's worth of code before building. The
order respects the dependency chain (`vk_context` → `simulation` → `report`
→ `main`) and mirrors `plan.md`.

For each file, the relevant lesson part tells you what every line does.

### Step 1 — Scaffolding (lesson 03)

1. `CMakeLists.txt` — minimal: `project`, C++17, `find_package(Vulkan)`,
   `find_program(GLSLC ...)`, the `.comp → .spv` custom command + `shaders`
   target, one `add_executable(profiler src/main.cpp)` linking
   `Vulkan::Vulkan` with the `SHADER_DIR` define.
2. `shaders/stub.comp` — the 3-line no-op, just to exercise the shader
   build.
3. `src/main.cpp` — temporary: `int main() { std::printf("scaffold ok\n"); }`

**Checkpoint:** `cmake -S . -B build && cmake --build build && ./build/profiler`
prints the line, and `build/shaders/stub.comp.spv` exists.

### Step 2 — Vulkan bring-up (lessons 02, 05)

4. `src/vk_context.h` — the `VulkanContext` struct + `loadShaderModule`
   declaration.
5. `src/vk_context.cpp` — the anonymous-namespace helpers (`debugCallback`,
   `hasLayer`, `hasInstanceExtension`, `check`), then `create`, `destroy`,
   `findMemoryType`, `deviceSummary`, `loadShaderModule`.
6. `src/main.cpp` — rewrite: create a `VulkanContext` with validation,
   print `deviceSummary()`, `destroy()`. Wrap in `try/catch`.
7. `CMakeLists.txt` — add `src/vk_context.cpp` to the executable.

**Checkpoint:** run prints a real device name and `Validation: enabled`,
with **zero** `[validation]` / VUID lines on stderr.

### Step 3 — Force shader + CPU cross-check (lessons 04, 06, 07)

8. `shaders/force_calc.comp` — the O(N²) softened-gravity shader.
9. `src/simulation.h` — first cut: `Vec3`, `Particle`, `SimParams`; a
   `Simulation` class with just `init` / `destroy` / `runForceCalcOnce` /
   `readAccelerations` and the buffer + pipeline + descriptor members; the
   free function `cpuForceCalc`. (No timestamps, no integrate, no batching
   yet.)
10. `src/simulation.cpp` — the file-local `check` + `PushConstants` +
    `alignUp`; `init` (command pool + `createBuffers` + `createPipelines`);
    `createBuffers` (the `makeBuffer` lambda, the DEVICE_LOCAL particle &
    accel buffers, the HOST_VISIBLE readback buffer, the staging upload);
    `createPipelines` (descriptor set layout, pipeline layout + push range,
    one `makePipeline` call for force, descriptor pool + set + writes);
    `beginCommands` / `submitAndWait`; `runForceCalcOnce`;
    `readAccelerations`; `destroy`; `cpuForceCalc`.
11. `src/main.cpp` — add `makeParticles`, `length` / `sub` / `relError`,
    `forceCalcCase`; `main` runs a few `forceCalcCase`s and reports
    PASS/FAIL.
12. `CMakeLists.txt` — add `src/simulation.cpp`.

**Checkpoint:** `worst rel-err ~1e-7  [PASS]` for several `(N, seed)`.

### Step 4 — Integration + momentum conservation (lessons 04, 07)

13. `shaders/integrate.comp` — semi-implicit Euler.
14. `src/simulation.h` — add `integratePipeline_`, `readParticles`,
    `stepOnce` (a single-step version for now — no timestamps, no batch),
    and the free function `totalMomentum`.
15. `src/simulation.cpp` — second `makePipeline` call for integrate; write
    the full one-step command buffer (force → compute/compute barrier →
    integrate → compute/transfer barrier → `vkCmdCopyBuffer` to the
    readback buffer → submit + wait); `readParticles`; `totalMomentum`.
16. `src/main.cpp` — add `momentumCase`; call it.

**Checkpoint:** `worst drift ~1e-6  [PASS]` over 100 steps.

### Step 5 — Timestamps (lessons 02 §12, 07, 11)

17. `src/simulation.h` — add `kTimestampsPerStep` / `kMaxBatch`, the
    `StepTimings` struct, `queryPool_` + `timestampsSupported_` +
    `timestampPeriod_` + `lastBatch_`, `createQueryPool`, `lastTimings()`,
    `timestampsSupported()`.
18. `src/simulation.cpp` — add `<chrono>` + `nanosBetween`;
    `createQueryPool`; weave into the step function: `vkCmdResetQueryPool`,
    six `vkCmdWriteTimestamp` calls at the stage boundaries, the three
    `Clock::now()` reads around submit/wait, `vkGetQueryPoolResults`, and
    filling one `StepTimings`.
19. `src/main.cpp` — add `timestampCase` (monotonic ticks, bounded
    durations, GPU span ≤ wall clock); call it.

**Checkpoint:** `timestamp instrumentation sanity ... [PASS]`.

### Step 6 — Report generator + `--report` (lesson 08, 09)

20. `src/report.h` — the `Report` class, `Stat` / `Means`, the nine
    vectors.
21. `src/report.cpp` — `addSample`, `summarize` (sort + mean/min/max +
    percentile lambda), `printTable`, `means`, `writeCsv`.
22. `src/main.cpp` — add `collectSteps`, `runReport`; the argument parser
    (`--report` / `--particles` / `--steps` / `--warmup` / `--csv` /
    `--help`); mode dispatch before the self-tests.
23. `CMakeLists.txt` — add `src/report.cpp`.

**Checkpoint:** `./build/profiler --report` prints the aligned table and
`stage sum vs GPU total ... 100.00% accounted`.

### Step 7 — Batching + sweep (lessons 07, 12)

24. `src/simulation.{h,cpp}` — generalise the single-step function into
    `runBatch(uint32_t steps)`: the per-step `for` loop, `base = 6*s`, the
    `last` flag and its last-vs-non-last barrier branch, amortising the CPU
    submit/wait cost by `/ steps`, filling `lastBatch_` with one
    `StepTimings` per step. Add `createLoopObjects` (reusable `loopCmd_` +
    `loopFence_` + persistent `readbackMapped_`) and switch the read
    functions to the persistent pointer. `stepOnce()` becomes
    `runBatch(1)`.
25. `src/report.{h,cpp}` — add `Means` + `means()` (if not already there).
26. `src/main.cpp` — add `--sweep` / `--batch`, `runSweep`,
    `batchEquivCase` and its self-test block; give `collectSteps` the
    `batch` parameter.

**Checkpoint:** `--sweep` emits one CSV row per N; the self-tests show
`batch=1 vs K  max |delta| = 0.000e+00  [PASS]`.

### Step 8 — Scripts (optional, no C++)

27. `scripts/run.sh` — the `DYLD_LIBRARY_PATH` shim.
28. `scripts/plot_sweep.py`, `scripts/plot_batching.py` — the charts
    (needs Python + matplotlib).

### Step 9 — The reusable layer (lesson 10) — fully independent

29. `layer/VkLayer_gcp_profiler.json.in` — the manifest template with the
    `@GCP_LAYER_LIB@` placeholder.
30. `layer/gcp_layer.cpp` — the dispatch-table structs, `dispatchKey`,
    `findInstLink` / `findDevLink`, `CreateInstance` / `CreateDevice` /
    `DestroyDevice`, the intercepted `vkCmd*` / `QueueSubmit` /
    wait functions, `GetInstanceProcAddr` / `GetDeviceProcAddr`,
    `vkNegotiateLoaderLayerInterfaceVersion`, `printReport`.
31. `CMakeLists.txt` — the `add_library(... SHARED)` target
    (headers-only, `CXX_VISIBILITY_PRESET hidden`) + the two-step
    `configure_file` / `file(GENERATE)` manifest generation.
32. `scripts/run_layer.sh` — sets `VK_ADD_LAYER_PATH` + `VK_INSTANCE_LAYERS`.

**Checkpoint:** `./scripts/run_layer.sh --report ...` prints the profiler's
own table *and* the layer's `=== VK_LAYER_gcp_profiler report ===` on exit,
with command counts that match (2 dispatch + 2 barrier + 1 copy per step)
and zero VUID.

### Rule of thumb while doing this

- Build after **every** file. A 20-line compile error is easy; a 200-line
  one is misery.
- Keep validation on from Step 2. Fix the first warning before writing the
  next line — they cascade.
- If a step feels too big, split it: get it *compiling* with function bodies
  that just `throw std::runtime_error("todo")`, then fill them in.

## The single most important sentence

If you remember nothing else:

> **This program runs a physics simulation on the GPU and measures how long
> each stage of GPU work takes — dispatch, synchronisation stall, memory
> copy — instead of just reporting one total number. Then it uses those
> measurements to find and fix a real performance bug.**
