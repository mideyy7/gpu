# Part 11 — Synchronisation and timestamps

This is the part interviewers probe hardest, because it's where "I used a
GPU API" separates from "I understand the GPU execution model." It expands
`docs/synchronisation.md`.

Two topics, tightly linked in this project:

1. **Pipeline barriers** — how you stop the GPU from running things in an
   order that corrupts data.
2. **`vkCmdWriteTimestamp`** — how you measure GPU time, and what the number
   actually means (which is subtler than it looks).

---

## 1. Why the GPU needs to be told to wait

A GPU is aggressively parallel and out-of-order. When you record:

```
vkCmdDispatch(force)       // writes accelBuffer
vkCmdDispatch(integrate)   // reads accelBuffer
```

...the GPU is **allowed to start `integrate` before `force` has finished**,
and it's allowed to start `integrate` before `force`'s writes to
`accelBuffer` have actually become visible in memory. If it does, `integrate`
reads stale or partially-written accelerations. Wrong physics, silently.

This is a **hazard**. The three classic kinds:

| Name | Shape | Example here |
|---|---|---|
| **RAW** (read-after-write) | B reads what A wrote | `integrate` reads the accel `force` wrote |
| **WAR** (write-after-read) | B overwrites what A still needs to read | next step's `integrate` writes positions this step's `force` was reading |
| **WAW** (write-after-write) | two writes to the same place, order matters | (not prominent here) |

Vulkan does **nothing** about hazards automatically (that's the "explicit"
in "explicit API"). You insert a **pipeline barrier** wherever a hazard
exists.

---

## 2. What a pipeline barrier actually says

`vkCmdPipelineBarrier` takes, among other things:

- a **source pipeline stage mask** (`srcStageMask`)
- a **destination pipeline stage mask** (`dstStageMask`)
- one or more **memory barriers**, each with a **source access mask**
  (`srcAccessMask`) and **destination access mask** (`dstAccessMask`)

It means:

> **Execution dependency:** all work in `srcStageMask` recorded *before* this
> barrier must reach that stage before any work in `dstStageMask` recorded
> *after* this barrier may start that stage.
>
> **Memory dependency:** the writes described by `srcAccessMask` (from that
> earlier work) are made *available and visible* to the accesses described
> by `dstAccessMask` (in that later work) — i.e. flushed from caches so the
> later reads see them.

### "Stage" vs "access" — the distinction interviewers want

- A **pipeline stage** is *when* in the GPU's processing pipeline something
  happens: `TOP_OF_PIPE` (the very start), `COMPUTE_SHADER` (running a
  compute shader), `TRANSFER` (a copy/blit), `BOTTOM_OF_PIPE` (the very
  end).
- An **access type** is *what kind of memory operation*: `SHADER_READ`,
  `SHADER_WRITE`, `TRANSFER_READ`, `TRANSFER_WRITE`.

You need both because "wait for the compute stage to finish" (execution) is
not the same as "and make sure its writes are actually in memory, not stuck
in a cache" (memory). A barrier with the right stages but empty access masks
would order execution but not guarantee the data is visible.

---

## 3. The three barriers in `runBatch`, precisely

Recall the per-step recording (Part 07). Each step has up to three
`memBarrier(...)` calls.

### Barrier 1 — after `force`, before `integrate` (every step)

```cpp
memBarrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
```

- src stage `COMPUTE_SHADER`, src access `SHADER_WRITE` — "the force
  shader's writes to `accelBuffer`."
- dst stage `COMPUTE_SHADER`, dst access `SHADER_READ` — "the integrate
  shader's reads of `accelBuffer`."
- Effect: integrate's compute stage won't start reading until force's
  compute writes are done **and visible**. Fixes the RAW hazard on
  `accelBuffer`.

### Barrier 2 (last step) — after `integrate`, before the readback copy

```cpp
memBarrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
           VK_PIPELINE_STAGE_TRANSFER_BIT);
```

- src `COMPUTE_SHADER` / `SHADER_WRITE` — integrate's writes to
  `particleBuffer`.
- dst `TRANSFER` / `TRANSFER_READ` — the `vkCmdCopyBuffer` that reads
  `particleBuffer`.
- The destination stage is `TRANSFER`, **not** `COMPUTE_SHADER`, because a
  buffer copy runs in the transfer stage. A common bug is to reuse
  `COMPUTE_SHADER` here; validation would flag the copy reading data with no
  dependency on the compute writes.

### Barrier 2 (non-last step) — after `integrate`, before next step's `force`

```cpp
memBarrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
```

Inside a batch, step `s`'s integrate writes `particleBuffer` and step
`s+1`'s force reads it. Same RAW hazard, same compute→compute barrier. **This
one barrier is what makes batching correct** — the step-to-step dependency
that a host round-trip (submit + fence wait) used to provide implicitly is
now an explicit in-queue barrier. Because the recorded commands are
otherwise identical, batched and unbatched runs produce bit-identical
results (the `batchEquivCase` test).

### Why these are `VkMemoryBarrier` (global), not `VkBufferMemoryBarrier`

The code uses a **global** memory barrier (covers all memory) rather than a
buffer-scoped one. It's slightly coarser than strictly necessary, but for a
pipeline this simple it's clear and correct, and the measured barrier cost
is tiny anyway. A buffer memory barrier scoped to exactly `accelBuffer` /
`particleBuffer` would be the "tighter" choice in a bigger renderer.

---

## 4. `vkCmdWriteTimestamp`: the subtle part

Naive mental model: "put a timestamp after the dispatch, it tells me when
the dispatch finished." **This is wrong**, and the wrongness matters for a
profiler.

`vkCmdWriteTimestamp(cmd, stage, pool, index)` records a command that writes
the GPU clock into `pool[index]` **when all previously-submitted commands
have reached `stage`**. It is *not* a barrier and does not, by itself, wait
for the previous dispatch to fully retire.

### Consequence 1 — the stage argument is load-bearing

- A timestamp with `VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT` after a dispatch can
  be satisfied almost instantly — "reached top of pipe" is trivially true
  for everything. So it's a good **start** marker (write it *before* the
  work) but a useless **end** marker.
- To close a bracket around a compute dispatch you write the timestamp at
  `VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT` (or `BOTTOM_OF_PIPE`). Even then it
  means "all prior work has passed the compute stage," not "this exact
  dispatch is done."

That's exactly why `runBatch` writes:

- `stamp(base+0, TOP_OF_PIPE)` — start of the step (write it *before* the
  force dispatch),
- `stamp(base+1..5, COMPUTE_SHADER or TRANSFER)` — each subsequent marker at
  the stage whose completion it's supposed to represent.

### Consequence 2 — the "barrier stall" number is the meaningful one

Without a barrier between two dispatches, they can overlap, so a timestamp
between them reflects that overlap rather than a clean hand-off. In
`runBatch` the "between" timestamps are placed *around the barriers*, and a
barrier is the only place the GPU is genuinely forced to stop. So
`ticks[2] - ticks[1]` (`barrier1Ns`) is the one number that answers "is my
synchronisation too coarse?"

If `barrier1Ns` were a large fraction of the frame, the fix would be to
narrow the barrier's scope (tighter stage/access masks, or a buffer-scoped
barrier). On this MoltenVK setup barriers are tens of microseconds and flat
in N, so the pipeline is *not* barrier-bound — but the instrumentation is
arranged so that if it were, you'd see it directly instead of having to
guess.

---

## 5. Ticks → nanoseconds

```cpp
span(a, b) = (ticks[b] - ticks[a]) * timestampPeriod_
```

- `ticks` are raw values of the GPU's timestamp counter.
- `timestampPeriod` (`VkPhysicalDeviceLimits::timestampPeriod`) is
  **nanoseconds per tick**. On MoltenVK it's `1.0` (tick = 1 ns). On other
  hardware it can be a fraction or a larger number.
- `timestampValidBits` (per queue family) says how many low bits of the
  counter are meaningful; `0` means timestamps aren't supported at all, in
  which case `createQueryPool` bails and every `StepTimings` comes back with
  `gpuValid = false`.

The reset step (`vkCmdResetQueryPool`) before writing is mandatory — reading
a slot that was written twice without a reset in between is undefined.

---

## 6. The batched-profiling caveat

When you record **many heavy dispatches back-to-back in one submission**
(large `--batch` *and* large `N`), the between-dispatch timestamps start to
absorb **queue-scheduling latency**: the GPU has a deep queue of work, and
"all prior work reached the compute stage" now includes waiting for a
backlog, not just this step's barrier.

So:

- `--batch 1` → accurate **per-stage** attribution (each step is its own
  submission, timestamps are clean).
- `--batch K` (K large) → the **frame-time-per-step** number stays
  meaningful (it's just total wall time / K), but the individual
  `barrier1Ns` / `integrateDispatchNs` numbers at large N get noisy.

This is documented in `docs/bottleneck.md` and is a good thing to volunteer
in an interview — it shows you understand that instrumentation changes what
you're measuring.

---

## 7. Fences vs. barriers vs. semaphores (don't mix these up)

| Primitive | Synchronises | Used here? |
|---|---|---|
| **Pipeline barrier** | work *within* one queue / command stream | yes, 3 per step |
| **Fence** | GPU → **CPU** ("the submission is done") | yes, `loopFence_` |
| **Semaphore** | GPU queue → GPU queue | no (one queue, one submission at a time) |
| **Event** | fine-grained within a command buffer | no |

A frequent interview question is "how does the CPU know the GPU finished?"
— the answer is **a fence**, signalled by `vkQueueSubmit` and waited on with
`vkWaitForFences`. Barriers are purely a GPU-internal ordering tool and
never involve the CPU.

---

## Explain it in one sentence

> Because the GPU runs work out of order and doesn't flush caches on its
> own, every read-after-write across `accelBuffer` and `particleBuffer` is
> guarded by a `vkCmdPipelineBarrier` with explicit source/destination
> pipeline stages (when) and access masks (what memory op), and the
> profiler's timestamps are placed around those barriers — with the start
> marker at `TOP_OF_PIPE` and the closing markers at `COMPUTE_SHADER` /
> `TRANSFER` — because `vkCmdWriteTimestamp` records "all prior work reached
> this stage," not "the previous command finished."

## Questions you should be able to answer

1. What is a RAW hazard, and name a specific one in this project.
2. In a pipeline barrier, what's the difference between the *stage* mask and
   the *access* mask, and why do you need both?
3. Why does the last-step barrier use destination stage `TRANSFER` while the
   non-last-step barrier uses `COMPUTE_SHADER`?
4. Which single barrier makes step batching correct, and why is the batched
   result bit-identical to the unbatched one?
5. Why is `vkCmdWriteTimestamp(TOP_OF_PIPE)` a fine *start* marker but a
   useless *end* marker?
6. What does `ticks[2] - ticks[1]` (`barrier1Ns`) actually measure, and what
   would you do if it were large?
7. How do raw ticks become nanoseconds? What is `timestampValidBits == 0`?
8. Why do per-stage numbers get noisy at large `--batch` and large `N`, but
   frame-time-per-step stays valid?
9. Fence, barrier, semaphore — which one tells the CPU the GPU is done?
