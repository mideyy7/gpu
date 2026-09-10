# Implementation Plan — GPU Compute Profiler (Vulkan)

## Platform note (read first)

This project is being built and tested on macOS/Apple Silicon, which has no
native Linux Vulkan driver. Vulkan is available locally only via **MoltenVK**
(Vulkan calls translated to Metal). The spec's own "what not to do" section
warns against MoltenVK precisely because the translation layer can skew
absolute timing numbers.

Resolution: build and validate every phase locally on MoltenVK — correctness,
validation-layer cleanliness, and the presence/shape of a stage-by-stage
breakdown are all fully verifiable this way. Every commit message and the
final README/write-up will flag which numbers came from MoltenVK, so nobody
mistakes translated-layer timings for native-driver timings. `CMakeLists.txt`
targets Vulkan generically (not Metal-specific) so the same code builds
against a native Linux driver unmodified if it's ever run there — only the
*headline numbers* would need re-capture on real hardware, not the code.

## Ground rules for every phase

1. Implement the phase's code.
2. Build it (`cmake --build build`).
3. Run the phase-specific test/check listed below — a phase is not "done"
   until its check passes, not until it merely compiles.
4. Fix anything the check surfaces before moving on. No phase starts until
   the previous one's check is green.
5. Commit with a message describing what the phase adds and how it was
   verified. Commits are authored solely as the repo owner — no co-author
   trailers.
6. Keep validation layers (`VK_LAYER_KHRONOS_validation`) enabled in debug
   builds from Phase 1 onward. Any validation error or warning is treated as
   a bug in that phase, not noise to silence later.

## Phase 0 — Scaffolding

- CMake project (C++17), directory layout: `src/`, `shaders/`, `tools/`
  (report generator), `docs/`.
- Fetch/link `Vulkan::Vulkan` via CMake's `find_package(Vulkan)`, GLFW is
  *not* needed (headless compute, no swapchain/window).
- Shader build step: compile `.comp` GLSL to SPIR-V via `glslc` as a CMake
  custom command, output into the build tree.
- `.gitignore` for `build/`.
- **Check:** `cmake -S . -B build && cmake --build build` succeeds producing
  an (empty/stub) executable.
- **Commit:** "Phase 0: project scaffolding, CMake + shader build wired up"

## Phase 1 — Vulkan instance & compute device bring-up

- Create `VkInstance` with validation layer + debug messenger enabled in
  debug builds.
- Enumerate physical devices, pick one with a compute-capable queue family.
- Create logical device + compute queue.
- Print selected device name, driver version, and whether it's a translation
  layer (MoltenVK reports itself in device properties) — this becomes the
  "measured on X" line the report/README will quote.
- **Check:** run the binary; it must print a real device name with **zero**
  validation errors/warnings on stderr. Verify by grepping run output for
  `VUID` (should be empty).
- **Commit:** "Phase 1: instance/device bring-up, clean under validation"

## Phase 2 — Particle buffers + force-calc compute shader (correctness first)

- Define particle struct (position, velocity, mass) and two SSBOs
  (ping/pong or in-place with barrier).
- Host-side init: N particles, small N (e.g. 4-16) with known deterministic
  initial state for the first correctness pass.
- Write `force_calc.comp`: naive O(N²) gravitational force accumulation.
- Single dispatch + fence wait + readback, no timestamps yet — get the math
  right before instrumenting it.
- Write a tiny CPU-side reference implementation of the same O(N²) force
  calc in the host app (or a standalone test), run it on the same initial
  state, and diff GPU output against CPU reference within a float epsilon.
- **Check:** automated diff (CPU reference vs GPU output) passes within
  epsilon for at least 3 different small-N seeded inputs. Validation layers
  still clean.
- **Commit:** "Phase 2: N-body force-calc compute shader, verified against
  CPU reference"

## Phase 3 — Integration stage + multi-step simulation loop

- Write `integrate.comp`: velocity/position update (semi-implicit Euler) as
  a second dispatch, separated from force-calc by a pipeline barrier
  (SSBO write→read hazard).
- Wire the full per-frame command buffer: dispatch(force) → barrier →
  dispatch(integrate) → barrier → copy to host-visible readback buffer.
- Run N steps; check a physical invariant that catches broken sync/math —
  total momentum conservation (drift bounded by epsilon) for a closed
  system with no external forces, checked every step for e.g. 100 steps.
- **Check:** momentum-drift assertion passes over the full run; validation
  layers still clean; still correct at the small-N used in Phase 2.
- **Commit:** "Phase 3: full sim loop (force+integrate+barriers), momentum
  conserved over N steps"

## Phase 4 — Timestamp instrumentation

- Create a `VkQueryPool` (timestamp type) sized for all boundaries per
  spec: before/after force dispatch, before/after each barrier, before/after
  integrate dispatch, at submit, at fence signal (CPU-side).
- Insert `vkCmdWriteTimestamp` at each GPU-side boundary; record
  `std::chrono` timestamps on the CPU side for submit and fence-signal wait.
- Read back query pool after fence signal, convert ticks to nanoseconds
  using `VkPhysicalDeviceLimits::timestampPeriod`.
- **Check:** dump raw per-stage tick/ns values for a run and manually sanity
  check: timestamps monotonically increase within a submission, no stage
  reports negative or absurd (>1s) duration, GPU-side sum ≤ wall-clock time
  measured on the CPU. Validation layers still clean (timestamp queries are
  a common source of new validation errors — pool reset ordering, etc.).
- **Commit:** "Phase 4: timestamp queries at all stage boundaries,
  sanity-checked"

## Phase 5 — Report generator

- Small standalone tool (can be a mode of the same binary, or a Python
  script reading a dumped CSV/JSON of raw timestamps — decide based on
  what's less friction once Phase 4's data shape is known).
- Turns raw per-boundary timestamps into a named per-stage breakdown:
  dispatch time (force), dispatch time (integrate), barrier/stall time,
  submission overhead, CPU-side wait time.
- Output as an aligned text table by default; this is the artifact that
  goes in the README's "sample output" section.
- **Check:** run generator on real captured data; every named stage sums to
  within rounding of the total frame time; spot-check by hand against the
  raw dump from Phase 4.
- **Commit:** "Phase 5: report generator, per-stage breakdown table"

## Phase 6 — MVP hardening

- CLI args: particle count, iteration count, output format.
- Confirm validation-clean end-to-end at a larger N (e.g. 4096+) — this is
  where synchronization bugs that don't show up at N=8 tend to surface.
- Write the real README (build/run instructions, sample output, the
  MoltenVK caveat from the top of this doc) and the short synchronization
  write-up.
- Record actual MVP success metrics (max stable particle count, per-stage
  breakdown for ≥3 stages) — no estimates, only measured numbers.
- **Check:** fresh clone + build + run following only the README succeeds.
- **Commit:** "Phase 6: MVP complete — README, write-up, measured success
  metrics"

**MVP ends here.** Everything below is stretch, attempted only after Phase 6
is fully green.

## Phase 7 (stretch) — Particle-count sweep + scaling chart

- Script to run the binary across a range of particle counts, collect
  per-stage breakdowns into one dataset.
- Render a stage-time-vs-N chart (dataviz skill for any chart output).
- **Check:** chart's shape matches expectation (O(N²) force-calc dominating
  at high N); numbers traceable back to raw per-run dumps.
- **Commit:** "Phase 7: particle-count sweep + scaling chart"

## Phase 8 (stretch) — Find and fix a real bottleneck

- Use Phase 7's data (and the Phase 5 report on individual runs) to name one
  concrete inefficiency — candidate suspects to check first: a barrier that
  could be split/scoped tighter, a readback that blocks the next dispatch
  instead of being pipelined, redundant queue submissions.
- Fix it, re-measure with the same methodology as before, report precise
  before/after numbers and a precisely-named root cause (not "made it
  faster").
- **Check:** before/after comparison run on the same hardware/params;
  correctness checks from Phase 3 (momentum conservation) still pass after
  the fix.
- **Commit:** "Phase 8: fix [specific bottleneck], before/after measured"

## Phase 9 (stretch) — Reusable Vulkan profiling layer

- Extract the Phase 4 instrumentation into a standalone `VK_LAYER` (layer
  JSON manifest + shared library implementing the relevant intercepted
  entry points: queue submit, command buffer record calls it needs to hook).
- Point the layer at the *existing* app binary (proving it works without
  modifying the profiled app) and confirm it produces an equivalent report
  to the baked-in instrumentation.
- **Check:** running the unmodified Phase-6 binary with
  `VK_INSTANCE_LAYERS=VK_LAYER_<name>` produces a report; diff against the
  Phase 5 in-app report for consistency.
- **Commit:** "Phase 9: profiling instrumentation packaged as a reusable
  Vulkan layer"

## Final wrap-up

- Update README with links to the scaling chart and the bottleneck
  before/after, and the finished resume bullet with real measured numbers.
- **Commit:** "Final: deliverables complete — chart, bottleneck fix, layer,
  write-up"
