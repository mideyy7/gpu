#pragma once

#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <vector>

#include "vk_context.h"

struct Vec3 {
    float x = 0, y = 0, z = 0;
};

// Matches the std430 layout of `struct Particle` in the compute shaders:
// vec4 pos_mass (xyz = position, w = mass), vec4 vel_pad (xyz = velocity).
struct Particle {
    float px = 0, py = 0, pz = 0, mass = 0;
    float vx = 0, vy = 0, vz = 0, _pad = 0;
};

struct SimParams {
    uint32_t particleCount = 0;
    float gravity = 1.0f;
    float softening = 0.1f;
    float dt = 0.01f;
};

// GPU-side N-body simulation. Phase 2: force-calc dispatch + accel readback.
class Simulation {
   public:
    void init(VulkanContext& ctx, const SimParams& params,
              const std::vector<Particle>& initial);
    void destroy();

    // Records force_calc, submits, waits on a fence. No instrumentation yet.
    void runForceCalcOnce();

    std::vector<Vec3> readAccelerations();

   private:
    void createBuffers(const std::vector<Particle>& initial);
    void createForceCalcPipeline();

    VulkanContext* ctx_ = nullptr;
    SimParams params_{};

    VkBuffer particleBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory particleMemory_ = VK_NULL_HANDLE;
    VkBuffer accelBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory accelMemory_ = VK_NULL_HANDLE;
    VkDeviceSize particleBytes_ = 0;
    VkDeviceSize accelBytes_ = 0;

    VkDescriptorSetLayout setLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline forceCalcPipeline_ = VK_NULL_HANDLE;

    VkCommandPool commandPool_ = VK_NULL_HANDLE;
};

// Reference CPU implementation of the same O(N^2) force accumulation, in the
// same j = 0..N-1 iteration order as the shader.
std::vector<Vec3> cpuForceCalc(const std::vector<Particle>& particles,
                               const SimParams& params);
