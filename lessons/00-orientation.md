# Part 00 — Orientation: what is this project?

No code in this part. The goal is that by the end you can describe the
project to a person and answer "why does it exist" and "what are the pieces".

---

## 1. The one-paragraph description

This is a command-line program. When you run it, it:

1. Talks to your **GPU** (graphics card / the graphics chip in your laptop).
2. Uploads a few thousand **particles** (points in 3D space with a position,
   a velocity, and a mass).
3. Repeatedly asks the GPU to run a **physics simulation step**: compute the
   gravitational pull every particle feels from every other particle, then
   move each particle a tiny bit.
4. While doing that, it **measures how long each part of the GPU's work
   takes** — the force calculation, the pauses where the GPU waits for its
   own memory writes to land, the copy of results back to the CPU.
5. It prints a **table** breaking the time down stage by stage.
6. It has extra modes that sweep across different particle counts, draw a
   chart, and demonstrate a real speed-up that was found using the tool
   itself.

The "profiler" part is the point. Anyone can print *one* number ("the frame
took 4 milliseconds"). This tool tells you **where** those 4 ms went, which
is what you need before you can make it faster.

---

## 2. Why these specific choices?

### Why Vulkan?

**Vulkan** is a low-level API (a set of C functions) for talking to GPUs.
"Low-level" means it hides almost nothing: you allocate the memory, you
decide when the GPU synchronises, you record the exact list of commands. The
opposite would be something like a game engine, where you call
`spawnParticles()` and the engine hides all of that.

For a *profiling* project, low-level is exactly what you want, because the
things you're measuring — dispatches, memory barriers, submissions — are
concepts Vulkan makes explicit. In a high-level API they're invisible, so
there'd be nothing to measure.

### Why an N-body particle simulation?

"N-body" means "N objects that all interact with each other." Gravity
between N particles is the classic example.

It was chosen over, say, multiplying two matrices because it naturally has
**several distinct stages**:

- **force calculation** — for each particle, sum the pull from all others,
- **integration** — use that force to update velocity and position,
- **readback** — copy the new positions from the GPU back to the CPU.

A profiler with only one stage to profile isn't a very convincing profiler.
This workload gives it three or four, separated by synchronisation points,
which is where the interesting measurements live.

It also has a **knob**: the particle count `N`. Turn it up and the force
calculation gets much heavier (it's O(N²) — doubling N roughly quadruples
the work). That lets you stress-test the profiler and study how each stage
scales.

### Why measure per stage instead of total time?

Because "make it faster" requires knowing what to attack. If the force
calculation is 90% of the time, optimising the memory copy is pointless. The
whole value proposition of the project is: *measure precisely, then act on
what you measured.* Part 12 is the payoff — a real bug found this way.

---

## 3. The shape of the codebase

Everything lives under one folder. Here's what each piece is for. Don't try
to understand the code yet — just learn the map.

```
shaders/
  force_calc.comp     The GPU program that computes gravity. Written in GLSL.
  integrate.comp      The GPU program that moves particles. Written in GLSL.
  stub.comp           A trivial placeholder from early scaffolding. Ignorable.

src/
  vk_context.{h,cpp}  Vulkan start-up: connect to the GPU, pick a device,
                      create the objects you need before you can do anything.
  simulation.{h,cpp}  The core. Owns the GPU memory buffers, the two shader
                      "pipelines", the timing query pool, and the loop that
                      records one simulation step (or a batch of steps),
                      submits it, waits, and reads back the timings.
  report.{h,cpp}      Takes the raw per-step timing numbers and turns them
                      into a human-readable table (mean / median / p99 / max
                      per stage) and a CSV file.
  main.cpp            The entry point. Parses command-line flags, then either
                      runs correctness self-tests, or a profiling run, or a
                      sweep across particle counts.

layer/
  gcp_layer.cpp       A *Vulkan layer* — a plugin that sits between any
                      Vulkan app and the driver and counts/times things,
                      without the app knowing. A "reusable" version of the
                      built-in instrumentation.

scripts/
  run.sh              Wrapper that sets an environment variable macOS needs.
  run_layer.sh        Same, but also activates the layer.
  plot_sweep.py       Reads a sweep CSV, draws the scaling chart.
  plot_batching.py    Draws the before/after chart for the bottleneck fix.

docs/
  synchronisation.md  Short write-up on one subtle Vulkan detail.
  bottleneck.md       The bug that was found and fixed, with numbers.
  *.svg / *.csv       Generated charts and data.

CMakeLists.txt        The build recipe: how to turn the source into a binary.
plan.md               The original phase-by-phase build plan.
README.md             Project overview for someone landing on the repo.
```

### The dependency direction

```
main.cpp
  ├── uses → report.{h,cpp}
  │            └── uses → simulation.h   (for the StepTimings struct)
  └── uses → simulation.{h,cpp}
               └── uses → vk_context.{h,cpp}
                            └── uses → Vulkan (the system library)

simulation.cpp  also loads → shaders/*.comp.spv   (compiled shader files)

layer/gcp_layer.cpp   is a *separate* build product. It does NOT use any of
                      the src/ files. It only uses Vulkan's headers.
```

So `vk_context` is the foundation, `simulation` builds on it, `report`
consumes what `simulation` produces, and `main` wires it all together. The
layer is off to the side, independent.

---

## 4. What "running it" looks like

```
$ ./scripts/run.sh --report --particles 8192 --steps 200

Device: Apple M1 (driver 0.2.2210, API 1.2.357) [translation layer: timings not native]
Validation: enabled

Per-stage timing  N=8192  batch=1/submit  device=Apple M1 ...
samples: 200 frames

stage                            mean        min        p50        p99        max   %frame
-------------------------- ---------- ---------- ---------- ---------- ---------- --------
force dispatch                3462.9us    2442.2us    3021.0us    7679.6us    8680.3us    76.4%
barrier 1 (accel w->r)          44.0us      16.5us      31.6us      60.2us    2630.4us     1.0%
integrate dispatch              33.7us      20.0us      34.3us      49.4us      67.2us     0.7%
barrier 2 (part. w->xfer)       34.2us      20.5us      34.1us      49.7us      55.2us     0.8%
readback copy                   32.5us      20.4us      32.5us      48.0us      50.8us     0.7%
-------------------------- ...
GPU total                     3607.3us    ...
submit call (CPU)              181.0us    ...
fence wait (CPU)              4240.9us    ...
frame total                   4531.5us    ...  100.0%
```

Read that as: "Over 200 simulation steps at 8192 particles, on average the
GPU spent 3463 µs on the force calculation, ~44 µs stalled at the first
barrier, and so on. The whole GPU part was 3607 µs; the CPU then spent 4241
µs waiting for the GPU to signal it was done; the total wall-clock time per
step was 4531 µs."

(`µs` = microsecond = one millionth of a second. 1000 µs = 1 ms.)

---

## 5. The four "modes" the program has

Set by command-line flags, handled in `main.cpp`:

| Command | What it does |
|---------|--------------|
| *(no flags)* | Runs **correctness self-tests** and prints PASS/FAIL. Proves the simulation is actually right before you trust its timings. |
| `--report` | Runs the sim for N steps and prints the **per-stage timing table** above. |
| `--sweep` | Runs the sim at many particle counts (256 … 32768) and prints **one CSV row per count**, for the scaling chart. |
| `--batch K` | (Modifier for the above.) Records **K simulation steps per GPU submission** instead of 1. This is the knob the bottleneck fix uses. |

---

## Explain it in one sentence

> It's a Vulkan compute application that runs an N-body particle simulation
> on the GPU and uses `vkCmdWriteTimestamp` queries to break GPU execution
> time down by pipeline stage, so you can see where time actually goes
> rather than reading a single wall-clock number — and it uses that
> breakdown to find and fix a real bottleneck.

## Questions you should be able to answer

1. Why Vulkan and not a game engine or CUDA?
2. Why an N-body simulation specifically, and not matrix multiplication?
3. What are the three or four "stages" the profiler measures?
4. What does `N` (particle count) control, and roughly how does the force
   calculation scale with it?
5. Name the five source areas (`vk_context`, `simulation`, `report`,
   `main`, `layer`) and one sentence on each.
6. What is the difference between the built-in instrumentation and the
   layer?
