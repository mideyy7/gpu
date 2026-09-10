# Part 13 — Interview preparation

You've read parts 00–12. This part is about *performing* that knowledge:
pitches at three lengths, the questions you'll actually get with answers,
and how to handle the weak spots.

---

## 1. Three pitches

### 30 seconds

> "It's a headless Vulkan compute app that runs an N-body particle
> simulation on the GPU and breaks GPU execution time down by pipeline
> stage — force dispatch, barrier stalls, integration, readback — using
> `vkCmdWriteTimestamp` queries, instead of reporting one wall-clock number.
> I used that breakdown to find and fix a real bottleneck: a fixed
> per-step submission round-trip that I removed by batching steps per
> submit, 4.2× faster in the regime where it dominated."

### 2 minutes

Add:

- **Why Vulkan:** it makes the execution model explicit — dispatches,
  barriers, submissions are all things you can put a timestamp around. A
  high-level API would hide exactly what I wanted to measure.
- **Why N-body:** it has several real stages (O(N²) force calc, integration,
  readback) separated by synchronisation, and one knob (particle count) to
  scale the workload.
- **Correctness first:** before trusting any timing, the GPU force math is
  cross-checked against a scalar CPU reference (~1e-7 error), and the full
  sim is checked for momentum conservation over 100 steps. Runs clean under
  the validation layer.
- **The finding:** `frame_total − gpu_total` was flat at ~0.5 ms regardless
  of N — a fixed cost. Not allocation (I tried hoisting the command
  buffer/fence, wash on this driver). It was the `vkQueueSubmit` +
  `vkWaitForFences` round-trip per step. Fixed by recording K steps per
  submit with a compute→compute barrier between steps. Bit-identical result,
  proven by a self-test.
- **The caveat:** measured via MoltenVK on an M1, so absolute submit/fence
  costs are inflated; the method holds, the headline number needs
  re-capturing on native hardware.

### 10 minutes

Walk the architecture: `vk_context` (bring-up) → `simulation` (`runBatch` is
the core: record dispatch/barrier/dispatch/barrier/copy, 6 timestamps, one
submit, one fence, CPU-time the submit and wait) → `report` (mean/p50/p99
per stage, reconciliation) → `main` (self-tests + `--report`/`--sweep`).
Then the sweep's two-regime finding (launch-bound vs O(N²) compute-bound),
then the bottleneck story in full, then the reusable layer (it independently
corroborated the fix: submits and copies 73→11 under `--batch 32`).

---

## 2. The questions you'll get, with answers

### "Why is the force calculation O(N²)?"

Every particle sums the gravitational pull from every other particle — an
N-outer, N-inner loop. Double N → 4× the pair interactions.

### "What is a pipeline barrier and where do you use one?"

An ordering + memory-visibility instruction to the GPU: work in the
destination stage recorded after the barrier can't start until work in the
source stage recorded before it finishes, and the source's writes are made
visible to the destination's reads. I use one between the force dispatch
(writes `accelBuffer`) and the integrate dispatch (reads it) — a
read-after-write hazard — with source stage/access `COMPUTE_SHADER` /
`SHADER_WRITE` and destination `COMPUTE_SHADER` / `SHADER_READ`. And a
second one before the readback copy, with destination stage `TRANSFER`
because a buffer copy runs in the transfer stage.

### "Stage mask vs access mask?"

Stage = *when* in the GPU pipeline (top-of-pipe, compute-shader, transfer).
Access = *what memory operation* (shader read, shader write, transfer read).
You need both: ordering execution isn't the same as guaranteeing the data is
flushed out of caches so the later stage sees it.

### "How does `vkCmdWriteTimestamp` work, and what's the gotcha?"

It records a command that writes the GPU clock when all prior work has
reached the stage you pass. The gotcha: it doesn't measure "the previous
command" — with `TOP_OF_PIPE` it fires almost immediately (good start
marker, useless end marker), and without a barrier between two dispatches
they overlap so a timestamp between them is muddy. That's why I place the
between-timestamps around the barriers, which are the only true
serialisation points.

### "How does the CPU know the GPU is finished?"

A fence. I pass `loopFence_` to `vkQueueSubmit`; the GPU signals it on
completion; the CPU blocks in `vkWaitForFences`; then `vkResetFences` to
reuse it. Barriers are GPU-internal and never involve the CPU; semaphores
are queue-to-queue and I don't need them with one queue.

### "Why DEVICE_LOCAL buffers with a staging copy instead of just
HOST_VISIBLE everywhere?"

DEVICE_LOCAL is the fast memory the shaders should touch; on a discrete GPU
the CPU can't map it at all. So the initial particles go through a small
HOST_VISIBLE staging buffer copied in once, and results come back via a
HOST_VISIBLE readback buffer written by an explicit `vkCmdCopyBuffer` — which
also makes the copy a real, measurable stage instead of hiding it.

### "How do you know the GPU results are correct?"

Two ways. `cpuForceCalc` is a scalar C++ reimplementation of the force
shader in the same loop order; the Phase 2 test runs both on identical
particles and checks agreement (~1e-7 relative). And momentum conservation —
`Σ mᵢvᵢ` must not drift because pairwise forces cancel — is checked every
step for 100 steps. Both under the validation layer with zero warnings.

### "Walk me through one simulation step."

(Recite the timeline from Part 07 §2 / Part 02 §13: reset+record the
command buffer, reset query pool, bind set + push constants, timestamp 0 at
top-of-pipe, bind+dispatch force, timestamp 1, barrier, timestamp 2,
bind+dispatch integrate, timestamp 3, barrier, timestamp 4, copy, timestamp
5, end, reset fence, `now()`, submit, `now()`, wait, `now()`, read the 6
ticks, subtract consecutive pairs for the 5 stage durations.)

### "How did you find the bottleneck?"

Computed `frame_total − gpu_total` across particle counts. It was flat at
~0.5 ms regardless of N — the signature of a fixed per-step cost. Ruled out
object allocation by hoisting the command buffer and fence (wash on this
driver). It was the submission round-trip: one `vkQueueSubmit` + one
blocking `vkWaitForFences` per step. Fixed by batching K steps per submit
with a compute→compute barrier between steps for the data dependency.

### "Why is batching safe?"

The recorded command stream for K batched steps is identical to K separate
single-step submissions — same dispatches, same barriers, grouped under one
submit. So the result is bit-identical, which a self-test asserts as
`max |Δ| == 0`, not "within tolerance."

### "Why doesn't batching help at large N?"

Above ~8192 particles the O(N²) force kernel is >90% of the frame; the ~1 ms
of overhead is under 2% of a 14 ms frame. Amortising it across 32 steps
saves ~1 ms total — noise. The win is real only where the round-trip
dominated the frame.

### "What would you do next?"

Two things. Re-run on native Linux + a real GPU — that de-risks the whole
premise and either confirms the bottleneck or shows part of it was MoltenVK.
And add a compute-bound optimisation: a tiled force kernel that stages
particle blocks into shared memory, which is the classic N-body speed-up and
would move the number at the sizes that actually matter.

### "What's the layer for?"

To show the instrumentation generalises. It's a Vulkan layer — a shared
library in the loader's call chain — that counts submits and commands and
CPU-times the submit and waits for *any* Vulkan app, no recompile. Running
the unmodified profiler under it, the command counts match the recorded
structure exactly and the CPU-wait number matches the in-app measurement
within a few percent, and `--batch 32` shows submits and copies dropping
73→11 — independent confirmation of the fix.

---

## 3. Owning the weak points

Interviewers respect "here's what's not great about this and why" far more
than a polished story with holes they have to find.

| Weak point | How to own it |
|---|---|
| **MoltenVK inflates the numbers** | Say it first, unprompted. "Absolute submit/fence costs are via a translation layer; the method is sound, the headline number needs native re-capture." |
| **The 4.2× is in the 'toy' regime** | "The win is where the round-trip dominated — small/medium N. Above 8192 it's a wash because the O(N²) kernel takes over. Stating that limit is the honest version." |
| **Simple workload (naive O(N²), no tiling)** | "Deliberately — I wanted several clean stages to profile, not a fast simulation. The obvious next step is a shared-memory tiled kernel." |
| **Small sample counts, noisy p99** | "Runs are a few hundred steps; I report p50 and p99 so the tail is visible, but I'd want longer runs and Release builds for publication numbers." |
| **No cross-check vs a real profiler** | "I validated internally — stage sums reconcile to GPU total, the layer corroborates the CPU numbers — but I didn't compare against RenderDoc/Streamline, which would be the next validation step." |
| **Manual `destroy()` instead of RAII** | "A deliberate readability trade-off — all teardown in one readable block. The idiomatic version wraps each handle in a type whose destructor calls `vkDestroyX`." |
| **Global memory barriers, not buffer-scoped** | "Coarser than strictly needed; correct and clear for a pipeline this small, and the measured barrier cost is negligible. A bigger renderer would scope them to the buffer." |

---

## 4. Things not to say

- Don't claim the numbers are native-GPU ground truth. They're MoltenVK.
- Don't say "it's fast" — it's a *profiler*, and the sim is deliberately
  naive. The point is measurement and the one fix.
- Don't overclaim the layer — it does counts + CPU timing, not GPU per-stage
  timing for arbitrary apps.
- Don't guess at Vulkan details you're unsure of. "I'd have to check the
  spec, but my understanding is…" is fine.

---

## 5. Codebase cheat-sheet (memorise this table)

| File | One line | Key thing to know |
|---|---|---|
| `shaders/force_calc.comp` | O(N²) softened gravity, one invocation per particle | `inversesqrt`, softening, `gl_GlobalInvocationID.x` |
| `shaders/integrate.comp` | semi-implicit Euler, in-place update | velocity updated first, then position |
| `src/vk_context.cpp` | instance → device → compute queue | conditional portability/validation = runs on MoltenVK *and* native |
| `src/simulation.cpp` `createBuffers` | 3 buffers + staging upload | DEVICE_LOCAL for shaders, HOST_VISIBLE readback, one staging buffer |
| `src/simulation.cpp` `createPipelines` | descriptors + 2 compute pipelines | one shared descriptor set (2 storage buffers), one shared pipeline layout, push constants |
| `src/simulation.cpp` `runBatch` | **the core**: record K steps, 1 submit, 1 fence, 6 timestamps/step | last step does the readback + a compute→transfer barrier; others do a compute→compute barrier |
| `src/report.cpp` | 9 parallel vectors → mean/p50/p99/max table | percentile = element at `p*(n-1)`; stage sum reconciles to GPU total |
| `src/main.cpp` | flags, 4 self-tests, `--report`/`--sweep` | force-vs-CPU, momentum, timestamp sanity, batch-equivalence |
| `layer/gcp_layer.cpp` | reusable layer: counts + CPU timing for any app | dispatch tables built in CreateInstance/CreateDevice, per-device state keyed by the handle's dispatch pointer |

---

## 6. A 60-second whiteboard diagram you can draw

```
 CPU                                          GPU
  |  record cmd buffer (dispatch/barrier/copy) |
  |  vkQueueSubmit  --------------------------> | force dispatch   [ts0..ts1]
  |  (time this)                               | barrier          [ts1..ts2]
  |  vkWaitForFences (BLOCK)                    | integrate        [ts2..ts3]
  |  (time this)                               | barrier          [ts3..ts4]
  |                                            | copy -> readback [ts4..ts5]
  |  <---------------- fence signalled --------|
  |  vkGetQueryPoolResults -> ts0..ts5         |
  |  stage durations = consecutive differences |
```

The bottleneck fix: instead of one `submit`/`wait` per step, record K steps
between `submit` and `wait`, with a barrier where each `wait` used to be.

---

## Explain it in one sentence

> I can pitch it at 30 seconds / 2 minutes / 10 minutes, answer the standard
> Vulkan questions (barrier stage-vs-access, `vkCmdWriteTimestamp`
> semantics, fence-vs-barrier-vs-semaphore, DEVICE_LOCAL + staging, how
> correctness is proven), walk one step end to end, tell the bottleneck
> story with the MoltenVK caveat volunteered, and name the weak points
> before I'm asked.

## Final checklist before an interview

- [ ] Can you draw the one-step timeline from memory?
- [ ] Can you say the bottleneck root cause in one sentence, mechanism
      included?
- [ ] Can you explain stage mask vs access mask without notes?
- [ ] Can you say why `vkCmdWriteTimestamp` "doesn't measure the previous
      command"?
- [ ] Can you state the MoltenVK caveat *and* why it doesn't sink the
      project?
- [ ] Can you name the next two things you'd do?
