# Part 12 — The bottleneck: finding it and fixing it

This is the payoff of the whole project, and the part an Arm interviewer
will care about most: *you measured something, understood it, changed it,
and measured again.* Source of record: `docs/bottleneck.md`.

---

## 1. How to read the report and spot a problem

Run at a middling particle count:

```
$ ./scripts/run.sh --report --particles 1024 --steps 200

stage                            mean   ...
force dispatch                 520.0us
barrier 1 ...                   33.0us
integrate dispatch              40.0us
barrier 2 ...                   45.0us
readback copy                   44.0us
GPU total                      726.0us
submit call (CPU)              120.0us
fence wait (CPU)              1108.0us
frame total                   1285.0us
```

Compute a derived number the table doesn't print directly:

```
overhead  =  frame_total  -  gpu_total
          =  1285  -  726
          =  559 us    <-- wall-clock time NOT spent doing GPU work
```

Now do it across several particle counts (this is what `--sweep` gives you):

| N | gpu_total | frame_total | overhead | overhead % of frame |
|---|-----------|-------------|----------|---------------------|
| 256 | 393 | 811 | 418 | 51% |
| 512 | 421 | 809 | 388 | 48% |
| 1024 | 726 | 1285 | 559 | 44% |
| 2048 | 1216 | 1957 | 741 | 38% |
| 4096 | 2338 | 3137 | 799 | 25% |
| 8192 | 3088 | 3848 | 760 | 20% |
| 16384 | 13038 | 14011 | 973 | 7% |
| 32768 | 51722 | 52892 | 1170 | 2% |

**The tell:** `overhead` is roughly **constant — about 0.4 to 1.0 ms —
regardless of N**. It doesn't scale with the workload. That's the signature
of a **fixed per-step cost**, and at small/medium N it's *half the frame*.

`submit call (CPU)` accounts for only ~50–320 µs of it. The rest is the
`fence wait` time *in excess of* the GPU's own timestamped span — i.e.
**queue-submission plumbing and fence-signal latency**, paid once for every
single step.

---

## 2. Why the loop had this cost

Before the fix, one step did exactly this on the CPU:

```
record a command buffer
vkQueueSubmit(...)          <-- hand work across to the driver/GPU
vkWaitForFences(...)        <-- BLOCK until the GPU signals back
read the results
... repeat, 200 times ...
```

Every step pays the **full CPU → driver → GPU → fence → CPU round-trip**.
On this MoltenVK/Metal setup that round-trip is ~0.4–0.5 ms and, being
fixed, it doesn't shrink as the workload grows. So at N ≤ 2048, where the
actual GPU compute is under a millisecond, you're spending as much time on
"are we there yet?" plumbing as on physics.

---

## 3. The fix that *didn't* move the needle (and why it stayed anyway)

The obvious first guess: the hot loop was allocating and freeing a command
buffer and creating and destroying a fence and mapping/unmapping the
readback buffer **every step**. Textbook "don't allocate in the render
loop."

So that was fixed — `createLoopObjects` (Part 06 §7): one reusable
`loopCmd_` (reset each step), one reusable `loopFence_` (`vkResetFences`
each step), one **persistent** mapping `readbackMapped_` held for the whole
run.

Result on MoltenVK: **within measurement noise.** The `overhead` numbers
barely moved. Conclusion: on this driver, allocating those small objects is
cheap; the cost is genuinely in the submit/signal path, which app-side
object reuse doesn't touch.

It was **kept anyway** because:

- it's correct and idiomatic Vulkan,
- on a *native* driver those per-step allocations *do* cost, so the code is
  better for it,
- and it's honest to say in an interview: "I tried the obvious
  micro-optimisation, measured, found it was a wash on this hardware because
  the bottleneck was elsewhere, kept it because it's right, and looked for
  the real lever."

That last sentence is worth more than the optimisation itself.

---

## 4. The fix that *did* work: batch steps per submit

**Root cause, stated precisely:**

> The simulation recorded and submitted one command buffer per step and
> blocked on its fence before recording the next, so every step paid the
> full, fixed CPU↔GPU submission + fence-signal round-trip (~0.5 ms on this
> MoltenVK setup, roughly independent of N).

**Fix:** `runBatch(K)` records **K simulation steps into one command
buffer** and issues **one `vkQueueSubmit` + one `vkWaitForFences`** for the
whole batch (Part 07). The step-to-step data dependency that the host
round-trip used to provide implicitly is expressed instead as a
`COMPUTE_SHADER → COMPUTE_SHADER`, `SHADER_WRITE → SHADER_READ` pipeline
barrier on the particle buffer between steps. The device→host readback copy
happens once, after the last step in the batch. Exposed as `--batch K`,
default 1.

**Why it's safe:** the recorded command stream for K batched steps is
*identical* to K separate single-step submissions — same dispatches, same
barriers, just grouped under one submit. So the result is **bit-identical**.
The `batchEquivCase` self-test asserts `max |Δ| == 0` on every particle
component for batch ∈ {16, 25, 32} vs batch 1. Not "within tolerance" —
exactly zero.

---

## 5. Before / after

Apple M1 via MoltenVK, 200 steps, mean frame time **per step**:

| N | batch = 1 | batch = 32 | speed-up | regime |
|---|-----------|-----------:|---------:|--------|
| 256 | 765 µs | 361 µs | **2.1×** | overhead-bound |
| 512 | 959 µs | 346 µs | **2.8×** | overhead-bound |
| 1024 | 1296 µs | 368 µs | **3.5×** | overhead-bound |
| 2048 | 1872 µs | 451 µs | **4.2×** | overhead-bound |
| 4096 | 2438 µs | 899 µs | 2.7× | mixed |
| 8192 | 3903 µs | 3451 µs | 1.13× | approaching compute-bound |
| 16384 | 14040 µs | 13544 µs | 1.04× | compute-bound |
| 32768 | 53224 µs | 52549 µs | 1.01× | compute-bound |

The fixed ~0.5 ms per-step round-trip, divided by 32, is ~16 µs per step
instead of ~500 µs. That's the whole mechanism.

`docs/batching.svg` shows this as two lines converging: batch-1 sits well
above batch-32 at small N, and they meet at large N.

---

## 6. Why the win is confined to small/medium N

Cross-reference the **scaling sweep** (Part 00 §2, `docs/scaling.svg`):

- **N ≲ 4096 — launch-overhead bound.** The force kernel is O(N²) but the
  GPU isn't saturated yet, so fixed per-dispatch/scheduling cost dominates.
  The per-step submit round-trip is a big fraction of the frame → batching
  helps a lot.
- **N ≳ 8192 — compute bound.** The O(N²) force kernel now dominates the
  frame (>90% at N=16384). The ~1 ms of overhead is <2% of a 14 ms frame.
  Batching amortises ~1 ms across 32 steps → saves ~1 ms total → invisible.

So the honest framing is: **"a 4.2× improvement in the regime where the
submission round-trip dominated (N ≤ 2048), and no change once the workload
is GPU-compute-bound — because at that point there's nothing left to
amortise."** Stating the limitation *is* the sophisticated answer.

---

## 7. The MoltenVK asterisk (say this before you're asked)

The entire story hinges on that ~0.5 ms round-trip. On MoltenVK that number
is inflated by the Vulkan→Metal translation layer. On a native Linux driver
the same round-trip might be 20–50 µs, which would shrink "4.2× at N=2048"
to something much smaller, or reveal that part of the cost was a MoltenVK
artifact.

That doesn't invalidate the *method* — measure overhead, see it's fixed in
N, attribute it, amortise it — but it does mean the headline number needs
re-capture on real hardware before quoting it externally. The code is
written to build and run unchanged there; only the numbers need redoing.
This caveat is in the README, `docs/bottleneck.md`, and the resume bullet.

---

## 8. How to tell this story in 90 seconds

> "I ran the profiler across particle counts and computed frame_total minus
> gpu_total — the wall-clock time not spent on GPU work. It was flat at
> about half a millisecond regardless of workload size, which is the
> signature of a fixed per-step cost. It wasn't object allocation — I tried
> hoisting the command buffer and fence out of the loop and it was a wash on
> this driver. It was the submission round-trip itself: one `vkQueueSubmit`
> plus one blocking `vkWaitForFences` per step. I changed the loop to record
> K steps into one command buffer with one submit, using a compute-to-compute
> barrier between steps instead of a host round-trip for the step
> dependency. Bit-identical results, verified by a self-test. That's 4.2×
> faster frame time at 2048 particles and basically no change above 8192,
> because past that the O(N²) force kernel dominates and there's nothing
> left to amortise. The absolute numbers are via MoltenVK so they'd need
> re-checking on native hardware, but the approach holds."

---

## Explain it in one sentence

> `frame_total − gpu_total` held flat at ~0.5 ms across all particle
> counts — a fixed per-step `vkQueueSubmit` + fence round-trip — so
> `runBatch(K)` records K steps per submit with an inter-step
> compute→compute barrier instead of a host round-trip, giving up to 4.2×
> faster per-step frame time in the overhead-bound regime (N ≤ 2048), no
> change once compute-bound, and a bit-identical result proven by a
> self-test.

## Questions you should be able to answer

1. What derived quantity did you look at to find the bottleneck, and what
   pattern in it was the tell?
2. What was the precise root cause? (Not "it was slow" — the mechanism.)
3. What fix did you try first, why did it not help on this hardware, and why
   did you keep it?
4. How does batching actually reduce per-step time — what's the arithmetic?
5. Why is the batched result bit-identical, and how is that verified?
6. Why does the speed-up vanish above N ≈ 8192?
7. What's the MoltenVK caveat on the headline number, and why doesn't it
   invalidate the method?
