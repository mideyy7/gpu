#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "report.h"
#include "simulation.h"
#include "vk_context.h"

namespace {

std::vector<Particle> makeParticles(uint32_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> pos(-5.0f, 5.0f);
    std::uniform_real_distribution<float> mass(0.5f, 2.0f);
    std::uniform_real_distribution<float> vel(-1.0f, 1.0f);
    std::vector<Particle> p(n);
    for (auto& q : p) {
        q.px = pos(rng);
        q.py = pos(rng);
        q.pz = pos(rng);
        q.mass = mass(rng);
        q.vx = vel(rng);
        q.vy = vel(rng);
        q.vz = vel(rng);
    }
    return p;
}

float length(const Vec3& v) { return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }
Vec3 sub(const Vec3& a, const Vec3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }

float relError(const Vec3& a, const Vec3& b) {
    return length(sub(a, b)) / (length(b) + 1e-4f);
}

// --- Phase 2: force-calc cross-check against CPU reference ---
bool forceCalcCase(VulkanContext& ctx, uint32_t n, uint32_t seed, float tol) {
    SimParams params;
    params.particleCount = n;
    params.gravity = 1.0f;
    params.softening = 0.5f;

    auto particles = makeParticles(n, seed);
    Simulation sim;
    sim.init(ctx, params, particles);
    sim.runForceCalcOnce();
    auto gpu = sim.readAccelerations();
    sim.destroy();

    auto cpu = cpuForceCalc(particles, params);
    float worst = 0.0f;
    int worstIdx = -1;
    for (uint32_t i = 0; i < n; ++i) {
        float e = relError(gpu[i], cpu[i]);
        if (e > worst) {
            worst = e;
            worstIdx = static_cast<int>(i);
        }
    }
    bool ok = worst <= tol;
    std::printf("  N=%-5u seed=%-3u  worst rel-err=%.3e @ i=%d  [%s]\n", n, seed, worst,
                worstIdx, ok ? "PASS" : "FAIL");
    return ok;
}

// --- Phase 3: momentum conservation over a multi-step run ---
bool momentumCase(VulkanContext& ctx, uint32_t n, uint32_t seed, uint32_t steps, float tol) {
    SimParams params;
    params.particleCount = n;
    params.gravity = 1.0f;
    params.softening = 0.5f;
    params.dt = 0.005f;

    auto particles = makeParticles(n, seed);
    const Vec3 p0 = totalMomentum(particles);
    const float p0len = length(p0);

    Simulation sim;
    sim.init(ctx, params, particles);

    float worstDrift = 0.0f;
    uint32_t worstStep = 0;
    for (uint32_t s = 1; s <= steps; ++s) {
        sim.stepOnce();
        auto state = sim.readParticles();
        float drift = length(sub(totalMomentum(state), p0));
        if (drift > worstDrift) {
            worstDrift = drift;
            worstStep = s;
        }
    }
    sim.destroy();

    float relDrift = worstDrift / (p0len + 1e-4f);
    bool ok = relDrift <= tol;
    std::printf("  N=%-5u seed=%-3u steps=%u  |P0|=%.4f  worst drift=%.3e (rel %.3e) "
                "@ step %u  [%s]\n",
                n, seed, steps, p0len, worstDrift, relDrift, worstStep,
                ok ? "PASS" : "FAIL");
    return ok;
}

// --- Phase 4: timestamp instrumentation sanity checks ---
bool timestampCase(VulkanContext& ctx, uint32_t n, uint32_t warmup, uint32_t seed) {
    SimParams params;
    params.particleCount = n;

    auto particles = makeParticles(n, seed);
    Simulation sim;
    sim.init(ctx, params, particles);

    if (!sim.timestampsSupported()) {
        std::printf("  timestamps unsupported on this queue -- skipping\n");
        sim.destroy();
        return true;
    }

    for (uint32_t i = 0; i < warmup; ++i) sim.stepOnce();
    sim.stepOnce();
    const StepTimings t = sim.lastTimings();
    sim.destroy();

    bool ok = true;
    if (!t.gpuValid) {
        std::printf("  [FAIL] GPU timestamps not readable\n");
        return false;
    }

    for (uint32_t i = 1; i < 6; ++i) {
        if (t.rawTicks[i] < t.rawTicks[i - 1]) {
            std::printf("  [FAIL] non-monotonic ticks at %u: %llu < %llu\n", i,
                        (unsigned long long)t.rawTicks[i],
                        (unsigned long long)t.rawTicks[i - 1]);
            ok = false;
        }
    }
    const double stages[5] = {t.forceDispatchNs, t.barrier1Ns, t.integrateDispatchNs,
                              t.barrier2Ns, t.copyNs};
    for (double s : stages) {
        if (s < 0.0 || s > 1e9) {
            std::printf("  [FAIL] stage duration out of range: %.1f ns\n", s);
            ok = false;
        }
    }
    if (t.gpuTotalNs > t.frameTotalNs * 1.05 + 1e5) {
        std::printf("  [FAIL] GPU total %.0f ns exceeds wall-clock frame %.0f ns\n",
                    t.gpuTotalNs, t.frameTotalNs);
        ok = false;
    }

    std::printf("  N=%-6u  force=%.1fus barrier1=%.1fus integrate=%.1fus "
                "barrier2=%.1fus copy=%.1fus\n",
                n, t.forceDispatchNs / 1e3, t.barrier1Ns / 1e3, t.integrateDispatchNs / 1e3,
                t.barrier2Ns / 1e3, t.copyNs / 1e3);
    std::printf("            gpuTotal=%.1fus  submitCall=%.1fus  cpuWait=%.1fus  "
                "frameTotal=%.1fus  [%s]\n",
                t.gpuTotalNs / 1e3, t.submitCallNs / 1e3, t.cpuWaitNs / 1e3,
                t.frameTotalNs / 1e3, ok ? "PASS" : "FAIL");
    return ok;
}

// --- Phase 5: profiling run + report ---
int runReport(VulkanContext& ctx, uint32_t particles, uint32_t steps, uint32_t warmup,
              const std::string& csvPath) {
    SimParams params;
    params.particleCount = particles;

    auto initial = makeParticles(particles, /*seed=*/1);
    const Vec3 p0 = totalMomentum(initial);

    Simulation sim;
    sim.init(ctx, params, initial);

    for (uint32_t i = 0; i < warmup; ++i) sim.stepOnce();

    Report report;
    for (uint32_t i = 0; i < steps; ++i) {
        sim.stepOnce();
        report.addSample(sim.lastTimings());
    }

    auto finalState = sim.readParticles();
    sim.destroy();

    char header[256];
    std::snprintf(header, sizeof(header), "Per-stage timing  N=%u  device=%s", particles,
                  ctx.deviceSummary().c_str());
    report.printTable(stdout, header);

    float drift = length(sub(totalMomentum(finalState), p0));
    std::printf("\ncorrectness: momentum drift after %u steps = %.3e (rel %.3e)\n", steps,
                drift, drift / (length(p0) + 1e-4f));

    if (!csvPath.empty()) {
        std::FILE* f = std::fopen(csvPath.c_str(), "w");
        if (!f) {
            std::fprintf(stderr, "error: cannot open %s for writing\n", csvPath.c_str());
            return EXIT_FAILURE;
        }
        report.writeCsv(f);
        std::fclose(f);
        std::printf("wrote per-step CSV: %s (%zu rows)\n", csvPath.c_str(),
                    report.sampleCount());
    }
    return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char** argv) {
    bool reportMode = false;
    uint32_t particles = 8192, steps = 200, warmup = 10;
    std::string csvPath;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "error: %s needs a value\n", a.c_str());
                std::exit(EXIT_FAILURE);
            }
            return argv[++i];
        };
        if (a == "--report")
            reportMode = true;
        else if (a == "--particles")
            particles = static_cast<uint32_t>(std::strtoul(next(), nullptr, 10));
        else if (a == "--steps")
            steps = static_cast<uint32_t>(std::strtoul(next(), nullptr, 10));
        else if (a == "--warmup")
            warmup = static_cast<uint32_t>(std::strtoul(next(), nullptr, 10));
        else if (a == "--csv")
            csvPath = next();
        else if (a == "--help" || a == "-h") {
            std::printf(
                "usage: profiler [--report] [--particles N] [--steps S] [--warmup W]\n"
                "                [--csv PATH]\n"
                "  no args : run correctness self-tests (phases 2-4)\n"
                "  --report: run the sim and print a per-stage timing breakdown\n");
            return EXIT_SUCCESS;
        } else {
            std::fprintf(stderr, "error: unknown argument '%s' (try --help)\n", a.c_str());
            return EXIT_FAILURE;
        }
    }

    VulkanContext ctx;
    try {
        ctx.create(/*enableValidation=*/true);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fatal: %s\n", e.what());
        return EXIT_FAILURE;
    }

    std::printf("Device: %s\n", ctx.deviceSummary().c_str());
    std::printf("Validation: %s\n", ctx.validationEnabled ? "enabled" : "disabled");

    if (reportMode) {
        int rc;
        try {
            rc = runReport(ctx, particles, steps, warmup, csvPath);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "fatal: %s\n", e.what());
            ctx.destroy();
            return EXIT_FAILURE;
        }
        ctx.destroy();
        return rc;
    }

    int failures = 0;
    try {
        std::printf("\nPhase 2: force-calc GPU vs CPU reference\n");
        const float forceTol = 2e-3f;
        const struct {
            uint32_t n, seed;
        } fcases[] = {{4, 1}, {8, 2}, {16, 3}, {13, 99}};
        for (const auto& c : fcases)
            if (!forceCalcCase(ctx, c.n, c.seed, forceTol)) ++failures;

        std::printf("\nPhase 3: momentum conservation over multi-step run\n");
        const float momTol = 5e-3f;
        const struct {
            uint32_t n, seed, steps;
        } mcases[] = {{8, 7, 100}, {64, 11, 100}, {256, 42, 100}};
        for (const auto& c : mcases)
            if (!momentumCase(ctx, c.n, c.seed, c.steps, momTol)) ++failures;

        std::printf("\nPhase 4: timestamp instrumentation sanity\n");
        const uint32_t tcounts[] = {256, 1024, 4096};
        for (uint32_t n : tcounts)
            if (!timestampCase(ctx, n, /*warmup=*/3, /*seed=*/5)) ++failures;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fatal: %s\n", e.what());
        ctx.destroy();
        return EXIT_FAILURE;
    }

    ctx.destroy();
    std::printf("\n%s (%d cases failed)\n", failures == 0 ? "ALL PASS" : "FAILURES",
                failures);
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
