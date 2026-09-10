# Part 09 — `main.cpp`: entry point, self-tests, run modes

File: `src/main.cpp` (416 lines).

**Job:** parse the command line, create the `VulkanContext`, then dispatch to
one of three things: the correctness self-tests (default), a profiling run
(`--report`), or a particle-count sweep (`--sweep`).

Everything except `main` itself lives in an anonymous `namespace { ... }`
(file-private).

---

## 1. Small helpers (lines 16-39)

### `makeParticles` (lines 16-32)

```cpp
std::vector<Particle> makeParticles(uint32_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> pos(-5.0f, 5.0f);
    std::uniform_real_distribution<float> mass(0.5f, 2.0f);
    std::uniform_real_distribution<float> vel(-1.0f, 1.0f);
    std::vector<Particle> p(n);
    for (auto& q : p) {
        q.px = pos(rng); q.py = pos(rng); q.pz = pos(rng);
        q.mass = mass(rng);
        q.vx = vel(rng);  q.vy = vel(rng);  q.vz = vel(rng);
    }
    return p;
}
```

Build `n` particles with pseudo-random positions in `[-5,5]³`, mass in
`[0.5,2]`, velocity in `[-1,1]³`. **Seeded** (`std::mt19937 rng(seed)`), so
the same `(n, seed)` always yields the same particles — that's what makes
the self-tests reproducible and lets `batchEquivCase` compare two runs from
identical input.

### `length`, `sub`, `relError` (lines 34-39)

```cpp
float length(const Vec3& v) { return std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z); }
Vec3 sub(const Vec3& a, const Vec3& b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
float relError(const Vec3& a, const Vec3& b) {
    return length(sub(a, b)) / (length(b) + 1e-4f);
}
```

Basic vector math the tests need. `relError` = "how far apart are `a` and
`b`, relative to the size of `b`" — the `+ 1e-4f` in the denominator stops a
near-zero `b` from producing a huge or infinite ratio.

---

## 2. The four self-test functions

Each returns `bool` (passed?), prints a one-line result, and is called in a
loop from `main` with a table of cases.

### `forceCalcCase` — Phase 2 (lines 42-69)

```cpp
bool forceCalcCase(VulkanContext& ctx, uint32_t n, uint32_t seed, float tol) {
    SimParams params;  params.particleCount = n; params.gravity = 1.0f; params.softening = 0.5f;
    auto particles = makeParticles(n, seed);

    Simulation sim;
    sim.init(ctx, params, particles);
    sim.runForceCalcOnce();
    auto gpu = sim.readAccelerations();
    sim.destroy();

    auto cpu = cpuForceCalc(particles, params);

    float worst = 0.0f; int worstIdx = -1;
    for (uint32_t i = 0; i < n; ++i) {
        float e = relError(gpu[i], cpu[i]);
        if (e > worst) { worst = e; worstIdx = (int)i; }
    }
    bool ok = worst <= tol;
    std::printf("  N=%-5u seed=%-3u  worst rel-err=%.3e @ i=%d  [%s]\n",
                n, seed, worst, worstIdx, ok ? "PASS" : "FAIL");
    return ok;
}
```

**"Is the GPU force math correct?"** Run one force dispatch on the GPU, run
`cpuForceCalc` on the same particles, compare acceleration per particle. The
worst relative error must be ≤ `tol` (2e-3 in `main`, actual values seen are
~1e-7). Prints which particle was worst.

### `momentumCase` — Phase 3 (lines 72-106)

```cpp
bool momentumCase(VulkanContext& ctx, uint32_t n, uint32_t seed, uint32_t steps, float tol) {
    // ... set params, make particles ...
    const Vec3 p0 = totalMomentum(particles);
    const float p0len = length(p0);

    Simulation sim;  sim.init(ctx, params, particles);

    float worstDrift = 0.0f; uint32_t worstStep = 0;
    for (uint32_t s = 1; s <= steps; ++s) {
        sim.stepOnce();
        auto state = sim.readParticles();
        float drift = length(sub(totalMomentum(state), p0));
        if (drift > worstDrift) { worstDrift = drift; worstStep = s; }
    }
    sim.destroy();

    float relDrift = worstDrift / (p0len + 1e-4f);
    bool ok = relDrift <= tol;
    std::printf("  ... worst drift=%.3e (rel %.3e) @ step %u  [%s]\n", ...);
    return ok;
}
```

**"Does the full simulation preserve a physical invariant?"** Record the
total momentum before, run `steps` steps (one at a time), and after each
step recompute momentum and track the largest deviation from the start. It
must stay within `tol` (5e-3; actual ~1e-6). This catches broken
synchronisation (a missing barrier → integrate reads stale accel → momentum
leaks) or broken math without needing a full reference simulator.

### `timestampCase` — Phase 4 (lines 109-165)

```cpp
bool timestampCase(VulkanContext& ctx, uint32_t n, uint32_t warmup, uint32_t seed) {
    // ... init sim ...
    if (!sim.timestampsSupported()) { /* skip, return true */ }

    for (uint32_t i = 0; i < warmup; ++i) sim.stepOnce();   // warm up
    sim.stepOnce();
    const StepTimings t = sim.lastTimings();
    sim.destroy();

    bool ok = true;
    // 1. ticks must be non-decreasing
    for (uint32_t i = 1; i < 6; ++i)
        if (t.rawTicks[i] < t.rawTicks[i-1]) { ...FAIL...; ok = false; }
    // 2. each stage duration in [0, 1e9] ns (i.e. 0..1 second)
    for (double s : {t.forceDispatchNs, t.barrier1Ns, ...}) 
        if (s < 0.0 || s > 1e9) { ...FAIL...; ok = false; }
    // 3. GPU span must fit inside the wall-clock frame
    if (t.gpuTotalNs > t.frameTotalNs * 1.05 + 1e5) { ...FAIL...; ok = false; }

    std::printf("  N=%-6u force=%.1fus ...  [%s]\n", ...);
    return ok;
}
```

**"Are the timestamp numbers sane?"** Not a value check — a *plausibility*
check:

1. The six raw ticks must be monotonically non-decreasing (time doesn't go
   backwards within a submission).
2. No stage duration is negative or absurd (> 1 second).
3. The GPU-measured span can't exceed the CPU wall-clock time for the step
   (plus 5% slack + 0.1 ms, to absorb clock-domain jitter).

`warmup` runs a few discarded steps first so the measured step isn't a cold
outlier (first submissions are slow — driver JIT, cache warm-up).

### `batchEquivCase` — Phase 8 (lines 168-200)

```cpp
bool batchEquivCase(VulkanContext& ctx, uint32_t n, uint32_t steps, uint32_t batch, uint32_t seed) {
    SimParams params; params.particleCount = n;
    auto initial = makeParticles(n, seed);

    auto runTo = [&](uint32_t b) {
        Simulation sim;  sim.init(ctx, params, initial);
        for (uint32_t done = 0; done < steps; ) {
            uint32_t k = std::min(b, steps - done);
            sim.runBatch(k);
            done += k;
        }
        auto out = sim.readParticles();
        sim.destroy();
        return out;
    };

    auto a = runTo(1);          // one step per submit
    auto c = runTo(batch);      // `batch` steps per submit

    float worst = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        worst = std::max(worst, std::fabs(a[i].px - c[i].px));
        worst = std::max(worst, std::fabs(a[i].py - c[i].py));
        worst = std::max(worst, std::fabs(a[i].pz - c[i].pz));
        worst = std::max(worst, std::fabs(a[i].vx - c[i].vx));
    }
    bool ok = worst == 0.0f;    // <-- EXACT equality expected
    std::printf("  ... max |delta| = %.3e  [%s]\n", ...);
    return ok;
}
```

**"Does batching change the answer?"** It must not. Run the same seeded
input twice — once with `batch=1`, once with `batch=K` — and compare the
final particle state. The command stream recorded for K batched steps is
*identical* to K separate single-step submissions (same dispatches, same
barriers, just grouped), so the result must be **bit-identical**: the check
is `worst == 0.0f`, not "within tolerance." This is the safety proof for the
Part 12 optimisation.

`runTo` is a lambda; the loop `for (done = 0; done < steps; )` advances by
`k = min(b, remaining)` so a `steps` that isn't a multiple of `b` still
finishes exactly.

---

## 3. `collectSteps` (lines 204-211)

```cpp
void collectSteps(Simulation& sim, uint32_t steps, uint32_t batch, Report& report) {
    for (uint32_t done = 0; done < steps; ) {
        uint32_t k = std::min(batch, steps - done);
        sim.runBatch(k);
        for (const auto& t : sim.lastBatchTimings()) report.addSample(t);
        done += k;
    }
}
```

Run `steps` total steps in chunks of `batch`, and after each `runBatch` feed
**every** per-step `StepTimings` it produced into the `Report`. So whether
you run `batch=1` or `batch=32`, the report ends up with the same number of
samples (one per step), just measured differently. Shared by `runReport` and
`runSweep`.

---

## 4. `runReport` — the `--report` mode (lines 214-255)

```cpp
int runReport(VulkanContext& ctx, uint32_t particles, uint32_t steps, uint32_t warmup,
              uint32_t batch, const std::string& csvPath) {
    SimParams params;  params.particleCount = particles;
    auto initial = makeParticles(particles, /*seed=*/1);
    const Vec3 p0 = totalMomentum(initial);

    Simulation sim;  sim.init(ctx, params, initial);

    for (uint32_t i = 0; i < warmup; ++i) sim.stepOnce();     // discard warm-up

    Report report;
    collectSteps(sim, steps, batch, report);

    auto finalState = sim.readParticles();
    sim.destroy();

    char header[288];
    std::snprintf(header, sizeof(header),
                  "Per-stage timing  N=%u  batch=%u/submit  device=%s",
                  particles, batch, ctx.deviceSummary().c_str());
    report.printTable(stdout, header);

    float drift = length(sub(totalMomentum(finalState), p0));
    std::printf("\ncorrectness: momentum drift after %u steps = %.3e (rel %.3e)\n",
                steps, drift, drift / (length(p0) + 1e-4f));

    if (!csvPath.empty()) {
        std::FILE* f = std::fopen(csvPath.c_str(), "w");
        if (!f) { std::fprintf(stderr, "error: cannot open %s ...\n", csvPath.c_str()); return EXIT_FAILURE; }
        report.writeCsv(f);
        std::fclose(f);
        std::printf("wrote per-step CSV: %s (%zu rows)\n", csvPath.c_str(), report.sampleCount());
    }
    return EXIT_SUCCESS;
}
```

Flow: make particles (fixed seed 1), record starting momentum, init the sim,
run `warmup` throwaway steps, then `collectSteps` for `steps` real steps
feeding the `Report`, read the final state, destroy the sim, print the table
with a header carrying `N`, batch size and device string, print a
correctness line (momentum drift over the whole run — an informational
sanity check, not a hard assertion here), and optionally dump the CSV.

Returns `EXIT_SUCCESS`/`EXIT_FAILURE` (only fails if the CSV file can't be
opened).

---

## 5. `runSweep` — the `--sweep` mode (lines 258-305)

```cpp
int runSweep(VulkanContext& ctx, uint32_t steps, uint32_t warmup, uint32_t batch,
             const std::string& csvPath) {
    const uint32_t counts[] = {256, 512, 1024, 2048, 4096, 8192, 16384, 32768};

    std::FILE* out = stdout;
    std::FILE* fileOut = nullptr;
    if (!csvPath.empty()) {
        fileOut = std::fopen(csvPath.c_str(), "w");
        if (!fileOut) { ...error...; return EXIT_FAILURE; }
        out = fileOut;
    }

    std::fprintf(out, "n,force_us,barrier1_us,integrate_us,barrier2_us,copy_us,"
                      "gpu_total_us,submit_us,cpu_wait_us,frame_total_us\n");
    std::fflush(out);

    for (uint32_t n : counts) {
        SimParams params;  params.particleCount = n;
        auto initial = makeParticles(n, /*seed=*/1);

        Simulation sim;  sim.init(ctx, params, initial);
        for (uint32_t i = 0; i < warmup; ++i) sim.stepOnce();
        Report r;
        collectSteps(sim, steps, batch, r);
        sim.destroy();

        const auto m = r.means();
        std::fprintf(out, "%u,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n", n,
                     m.force/1e3, m.barrier1/1e3, m.integrate/1e3, m.barrier2/1e3,
                     m.copy/1e3, m.gpuTotal/1e3, m.submitCall/1e3, m.cpuWait/1e3,
                     m.frameTotal/1e3);
        std::fflush(out);
        std::fprintf(stderr, "  N=%-6u frame=%.1fus force=%.1fus\n", n, m.frameTotal/1e3, m.force/1e3);
    }

    if (fileOut) { std::fclose(fileOut); std::fprintf(stderr, "wrote sweep CSV: %s\n", csvPath.c_str()); }
    return EXIT_SUCCESS;
}
```

For each particle count in a fixed list, build a fresh `Simulation`, warm
up, run `steps` steps, take `r.means()`, and write **one CSV row** of nine
means (converted ns → µs). The CSV goes to a file if `--csv` was given, else
to stdout; a short progress line always goes to stderr. `std::fflush` after
each row so you see progress even mid-run.

The output CSV is what `scripts/plot_sweep.py` turns into `docs/scaling.svg`.

---

## 6. `main` (lines 309-416)

### Argument parsing (lines 309-350)

```cpp
int main(int argc, char** argv) {
    bool reportMode = false, sweepMode = false;
    uint32_t particles = 8192, steps = 200, warmup = 10, batch = 1;
    std::string csvPath;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "error: %s needs a value\n", a.c_str()); std::exit(EXIT_FAILURE); }
            return argv[++i];
        };
        if      (a == "--report")    reportMode = true;
        else if (a == "--sweep")     sweepMode  = true;
        else if (a == "--particles") particles  = (uint32_t)std::strtoul(next(), nullptr, 10);
        else if (a == "--steps")     steps      = (uint32_t)std::strtoul(next(), nullptr, 10);
        else if (a == "--warmup")    warmup     = (uint32_t)std::strtoul(next(), nullptr, 10);
        else if (a == "--batch")     batch      = (uint32_t)std::strtoul(next(), nullptr, 10);
        else if (a == "--csv")       csvPath    = next();
        else if (a == "--help" || a == "-h") { std::printf("usage: ...\n"); return EXIT_SUCCESS; }
        else { std::fprintf(stderr, "error: unknown argument '%s' (try --help)\n", a.c_str()); return EXIT_FAILURE; }
    }
```

A hand-rolled parser. `argv` is an array of C strings; loop from index 1
(skip the program name). `next()` is a lambda that returns the *following*
argument and advances `i` — used for flags that take a value (`--particles
8192`). `std::strtoul(str, nullptr, 10)` parses a base-10 unsigned long from
a string. Unknown flag → error and exit. Defaults are set at the top
(`particles = 8192`, etc.).

### Context creation (lines 352-361)

```cpp
    VulkanContext ctx;
    try {
        ctx.create(/*enableValidation=*/true);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fatal: %s\n", e.what());
        return EXIT_FAILURE;
    }

    std::printf("Device: %s\n", ctx.deviceSummary().c_str());
    std::printf("Validation: %s\n", ctx.validationEnabled ? "enabled" : "disabled");
```

Create the Vulkan context with validation on; any exception from deep
inside becomes `fatal: <message>` and a non-zero exit. Then print the device
line (with the MoltenVK caveat if applicable) and whether validation
actually engaged.

### Mode dispatch (lines 363-375)

```cpp
    if (reportMode || sweepMode) {
        int rc;
        try {
            rc = sweepMode ? runSweep(ctx, steps, warmup, batch, csvPath)
                           : runReport(ctx, particles, steps, warmup, batch, csvPath);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "fatal: %s\n", e.what());
            ctx.destroy();
            return EXIT_FAILURE;
        }
        ctx.destroy();
        return rc;
    }
```

If either run mode was requested, do that one and exit. (`--sweep` wins if
both were somehow passed.) Note `ctx.destroy()` is called on both the
success and exception paths — Vulkan cleanup always happens.

### The self-tests (lines 377-415)

```cpp
    int failures = 0;
    try {
        std::printf("\nPhase 2: force-calc GPU vs CPU reference\n");
        const float forceTol = 2e-3f;
        const struct { uint32_t n, seed; } fcases[] = {{4,1},{8,2},{16,3},{13,99}};
        for (const auto& c : fcases)
            if (!forceCalcCase(ctx, c.n, c.seed, forceTol)) ++failures;

        std::printf("\nPhase 3: momentum conservation over multi-step run\n");
        const float momTol = 5e-3f;
        const struct { uint32_t n, seed, steps; } mcases[] = {{8,7,100},{64,11,100},{256,42,100}};
        for (const auto& c : mcases)
            if (!momentumCase(ctx, c.n, c.seed, c.steps, momTol)) ++failures;

        std::printf("\nPhase 4: timestamp instrumentation sanity\n");
        const uint32_t tcounts[] = {256, 1024, 4096};
        for (uint32_t n : tcounts)
            if (!timestampCase(ctx, n, /*warmup=*/3, /*seed=*/5)) ++failures;

        std::printf("\nPhase 8: step batching preserves the simulation result\n");
        const struct { uint32_t n, steps, batch; } bcases[] = {{64,96,16},{512,64,32},{2048,50,25}};
        for (const auto& c : bcases)
            if (!batchEquivCase(ctx, c.n, c.steps, c.batch, /*seed=*/3)) ++failures;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fatal: %s\n", e.what());
        ctx.destroy();
        return EXIT_FAILURE;
    }

    ctx.destroy();
    std::printf("\n%s (%d cases failed)\n", failures == 0 ? "ALL PASS" : "FAILURES", failures);
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
```

Four blocks, one per "phase," each a small table of anonymous-struct cases
run in a loop. `failures` counts any `false` return. At the end, print
`ALL PASS` or `FAILURES` and return 0 or 1 accordingly — so a CI script or
`echo $?` can tell whether the build is healthy.

The anonymous-struct-array pattern (`const struct { uint32_t n, seed; }
fcases[] = {{4,1}, ...};`) is a compact way to write "here are the test
inputs" inline.

---

## Explain it in one sentence

> `main` parses flags into `{reportMode, sweepMode, particles, steps,
> warmup, batch, csvPath}`, creates the validated `VulkanContext`, then
> either runs `runReport` / `runSweep`, or (default) runs four blocks of
> self-tests — force-vs-CPU, momentum conservation, timestamp sanity, and
> batch equivalence — counting failures and returning a non-zero exit code
> if any fail.

## Questions you should be able to answer

1. What are the four self-tests and what does each prove?
2. Why does `batchEquivCase` check `worst == 0.0f` (exact) while
   `forceCalcCase` uses a tolerance?
3. What does `warmup` do and why is it needed for the timing tests?
4. What does `collectSteps` do, and why does the report end up with the same
   sample count regardless of batch size?
5. How does the argument parser handle a flag that takes a value, like
   `--particles 8192`?
6. What determines the program's exit code in self-test mode, and why does
   that matter?
7. Where does the sweep CSV go, and what reads it afterward?
