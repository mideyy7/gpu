#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

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

}  // namespace

int main() {
    VulkanContext ctx;
    try {
        ctx.create(/*enableValidation=*/true);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fatal: %s\n", e.what());
        return EXIT_FAILURE;
    }

    std::printf("Device: %s\n", ctx.deviceSummary().c_str());
    std::printf("Validation: %s\n", ctx.validationEnabled ? "enabled" : "disabled");

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
