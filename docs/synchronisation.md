# One non-obvious thing about Vulkan's timestamp model

## `vkCmdWriteTimestamp` does not measure "the previous command"

The intuitive mental model — "put a timestamp after the dispatch and it tells
me when the dispatch finished" — is wrong in a way that matters for a
profiler.

`vkCmdWriteTimestamp(cmd, stage, pool, query)` writes the timestamp *once all
previously-submitted commands have reached `stage`*. It is not a barrier and
it does not, by itself, wait for the previous dispatch to *retire*. Two
consequences:

1. **The stage argument is load-bearing.** A timestamp written with
   `VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT` after a dispatch can be satisfied
   almost immediately — before the dispatch has done any work — because
   "reached top of pipe" is trivially true. To bracket a compute dispatch you
   want the *closing* timestamp at `VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT` (or
   `BOTTOM_OF_PIPE`), and even then you are measuring "all prior work has
   passed the compute stage", not "this specific dispatch is done".

2. **Without a barrier between them, adjacent dispatches overlap**, and the
   timestamp between them reports a value that reflects that overlap rather
   than a clean hand-off. The GPU is free to begin the second dispatch before
   the first drains unless a pipeline barrier creates the dependency.

This is why the profiler here places its "between" timestamps *around the
pipeline barriers*, not just after the dispatches. The barrier is what
actually serialises force-calc → integrate (the second kernel reads the
acceleration buffer the first one wrote), and the timestamps on either side
of it are what let the barrier's cost — the stall while the write is made
visible to the subsequent read — show up as its own line in the breakdown
instead of being smeared into the neighbouring dispatch times.

## The corollary: the "barrier stall" number is the useful one

Because a barrier is the only place in this pipeline where the GPU is forced
to stop and wait, the delta across it is the one number that answers "is my
synchronisation too coarse?". If `barrier 1` were a large fraction of the
frame, the fix would be to narrow its scope: a `VkMemoryBarrier` with
`srcStageMask = COMPUTE_SHADER` / `dstStageMask = COMPUTE_SHADER` and
`SHADER_WRITE → SHADER_READ` access masks is already fairly tight, but a
full `VK_PIPELINE_STAGE_ALL_COMMANDS_BIT` barrier (a common copy-paste
mistake) would serialise far more than the accel-buffer hazard requires and
would be visible here as an inflated barrier line.

On this MoltenVK setup the barrier lines are tens of microseconds and flat in
N, so the pipeline is not barrier-bound — the frame is bound on the O(N²)
force kernel and, above that, on the CPU blocking on the submission fence
every step. But the instrumentation is arranged so that *if* a barrier were
the problem, you would see it directly rather than having to infer it.
