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

// Relative error with an absolute floor, so near-zero components don't blow up.
float relError(const Vec3& a, const Vec3& b) {
    float num = std::sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) +
                          (a.z - b.z) * (a.z - b.z));
    float den = std::sqrt(b.x * b.x + b.y * b.y + b.z * b.z) + 1e-4f;
    return num / den;
}

bool runCase(VulkanContext& ctx, uint32_t n, uint32_t seed, float tol) {
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
    if (!ok && worstIdx >= 0) {
        const auto& g = gpu[worstIdx];
        const auto& c = cpu[worstIdx];
        std::printf("    gpu=(% .6f % .6f % .6f)\n    cpu=(% .6f % .6f % .6f)\n", g.x, g.y,
                    g.z, c.x, c.y, c.z);
    }
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
    std::printf("Validation: %s\n\n", ctx.validationEnabled ? "enabled" : "disabled");
    std::printf("Phase 2: force-calc GPU vs CPU reference\n");

    const float tol = 2e-3f;  // GPU inversesqrt vs CPU 1/sqrt precision gap
    struct Case {
        uint32_t n, seed;
    };
    const Case cases[] = {{4, 1}, {8, 2}, {16, 3}, {13, 99}};

    int failures = 0;
    try {
        for (const auto& c : cases)
            if (!runCase(ctx, c.n, c.seed, tol)) ++failures;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fatal: %s\n", e.what());
        ctx.destroy();
        return EXIT_FAILURE;
    }

    ctx.destroy();

    std::printf("\n%s (%d/%zu cases failed)\n", failures == 0 ? "ALL PASS" : "FAILURES",
                failures, sizeof(cases) / sizeof(cases[0]));
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
