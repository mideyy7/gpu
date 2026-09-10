# GPU Compute Profiler (Vulkan)

A headless Vulkan compute application that runs an N-body particle simulation
and breaks down GPU execution time **by pipeline stage** — force-calc dispatch,
pipeline-barrier stalls, integration dispatch, and device→host readback — using
custom `vkCmdWriteTimestamp` instrumentation rather than reporting a single
wall-clock number.

## What it does

Each simulation step records one command buffer:

```
[ts] force_calc dispatch  (O(N^2) softened gravity -> per-particle accel)
[ts] barrier   (accel: shader write -> shader read)
[ts] integrate dispatch   (semi-implicit Euler: v += a*dt; x += v*dt)
[ts] barrier   (particles: shader write -> transfer read)
[ts] copy      (device-local particle buffer -> host-visible readback buffer)
[ts]
```

Timestamps are written at every boundary. The CPU also times the
`vkQueueSubmit` call and the `vkWaitForFences` wait. After each step the
timestamp query pool is read back and converted to nanoseconds via
`VkPhysicalDeviceLimits::timestampPeriod`. The report generator aggregates
these across many steps into a mean / min / p50 / p99 / max table with each
stage's share of frame time.

## Platform note — timings here are via MoltenVK

This was built and measured on macOS / Apple Silicon (M1), where the only
Vulkan implementation available is **MoltenVK**, which translates Vulkan to
Metal. That translation layer sits between the timestamps and the real
hardware, so **absolute numbers below are indicative, not native-driver
ground truth** — barrier and submit costs in particular reflect MoltenVK's
emulation. The tool detects and prints this (`[translation layer: timings
not native]`).

The code targets Vulkan generically (`find_package(Vulkan)`, no Metal-specific
calls) and builds unmodified against a native Linux driver — only the headline
numbers would need re-capture there.

## Build

Requires: CMake ≥ 3.20, a C++17 compiler, the Vulkan SDK / loader, `glslc`,
and the Khronos validation layer.

macOS (Homebrew):

```
brew install vulkan-headers vulkan-loader vulkan-tools \
             vulkan-validationlayers molten-vk shaderc
cmake -S . -B build
cmake --build build
```

Linux (LunarG SDK on PATH): the same two `cmake` commands.

## Run

```
./scripts/run.sh                          # correctness self-tests (see below)
./scripts/run.sh --report                 # profiling run, default N=8192
./scripts/run.sh --report --particles 16384 --steps 300 --warmup 30
./scripts/run.sh --report --particles 8192 --csv run.csv   # per-step CSV
```

`scripts/run.sh` just sets `DYLD_LIBRARY_PATH` so the loader finds the
Homebrew validation-layer dylib on macOS; on Linux you can run
`./build/profiler` directly.

### Self-tests (no arguments)

Runs three correctness checks and exits non-zero on any failure:

- **force-calc vs CPU reference** — GPU acceleration output diffed against a
  scalar CPU implementation in the same summation order, for several seeded
  inputs including non-multiple-of-workgroup counts. Worst relative error
  observed ~1e-7.
- **momentum conservation** — total linear momentum `Σ mᵢvᵢ` must stay
  constant (pairwise forces cancel) across 100 steps for N = 8, 64, 256.
  Worst relative drift observed ~1e-6.
- **timestamp sanity** — ticks monotonic within a submission, every stage
  duration in [0, 1 s], GPU span ≤ wall-clock frame, for N up to 4096.

All three pass with **zero Vulkan validation errors or warnings**.

## Sample output

`./scripts/run.sh --report --particles 16384 --steps 300 --warmup 30`
(Apple M1 via MoltenVK):

```
stage                            mean        min        p50        p99        max   %frame
-------------------------- ---------- ---------- ---------- ---------- ---------- --------
force dispatch               13682.4us   11298.5us   13159.9us   20025.0us   20477.8us    92.4%
barrier 1 (accel w->r)          29.5us      17.2us      29.5us      45.9us      50.8us     0.2%
integrate dispatch              47.2us      22.4us      44.2us      79.8us      85.4us     0.3%
barrier 2 (part. w->xfer)       30.4us      17.7us      30.5us      48.5us      55.6us     0.2%
readback copy                   31.0us      17.8us      30.7us      51.2us      58.2us     0.2%
-------------------------- ---------- ---------- ---------- ---------- ---------- --------
GPU total                    13820.6us   11452.9us   13289.3us   20153.0us   20641.0us    93.3%
submit call (CPU)              243.1us      56.2us     231.1us     440.4us     490.3us     1.6%
fence wait (CPU)             14407.8us   12166.0us   13863.4us   20655.1us   21144.6us    97.3%
frame total                  14812.3us   12528.5us   14293.0us   20986.8us   21476.8us   100.0%
```

## Measured results (Apple M1 via MoltenVK)

| N       | force dispatch (mean) | frame total (mean) | vs. 2× N |
|---------|-----------------------|--------------------|----------|
| 16384   | 13.7 ms               | 14.8 ms            | —        |
| 32768   | 52.7 ms               | 54.0 ms            | 3.8×     |
| 65536   | 209.9 ms              | 211.1 ms           | 4.0×     |

- **Force-calc dominates**: 92–99% of frame time, and scales ~4× per 2× N —
  the expected O(N²) of the all-pairs kernel.
- **Barriers, integration, and readback are each <1% of frame** at these
  sizes (tens of µs, roughly flat in N) — but note this is MoltenVK barrier
  emulation, not a native driver.
- **The frame is serialised on the fence**: `fence wait` ≈ 97% of frame
  because every step submits one command buffer and blocks on it. This is
  the first thing a real optimisation pass would target (overlap the next
  step's recording / readback with GPU execution).
- Stable per-step frame time up to at least **N = 16384 at ~15 ms/step**;
  larger N runs correctly but single-precision force summation accumulates
  visible momentum drift over hundreds of Euler steps (expected, not a bug —
  the exact-arithmetic conservation is checked at moderate N in the
  self-tests).

## Repository layout

```
shaders/       force_calc.comp, integrate.comp  (GLSL -> SPIR-V via glslc)
src/
  vk_context.*   instance / device / queue bring-up, validation, memory types
  simulation.*   buffers, pipelines, barriers, timestamp query pool, step loop
  report.*       cross-step aggregation + table / CSV output
  main.cpp       arg parsing, self-tests, --report mode
scripts/run.sh   runtime env shim for macOS
docs/            synchronisation write-up
plan.md          phase-by-phase build plan
```

## See also

`docs/synchronisation.md` — one non-obvious thing about Vulkan's timestamp /
synchronisation model that this project surfaced.
