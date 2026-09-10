#pragma once

#include <vulkan/vulkan.h>

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
    float softening = 0.5f;
    float dt = 0.005f;
};

// GPU-side N-body simulation.
//   force_calc  -> barrier -> integrate -> barrier -> copy to readback buffer
// Particle/accel storage is DEVICE_LOCAL; a separate HOST_VISIBLE readback
// buffer receives an explicit vkCmdCopyBuffer so the copy is a real,
// measurable pipeline stage.
class Simulation {
   public:
    void init(VulkanContext& ctx, const SimParams& params,
              const std::vector<Particle>& initial);
    void destroy();

    // force_calc + barrier + copy(accel -> readback). No integration.
    void runForceCalcOnce();
    // Full step: force + barrier + integrate + barrier + copy(particles -> readback).
    void stepOnce();

    std::vector<Vec3> readAccelerations();
    std::vector<Particle> readParticles();

   private:
    void createBuffers(const std::vector<Particle>& initial);
    void createPipelines();
    VkCommandBuffer beginCommands();
    void submitAndWait(VkCommandBuffer cmd);

    VulkanContext* ctx_ = nullptr;
    SimParams params_{};

    VkBuffer particleBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory particleMemory_ = VK_NULL_HANDLE;
    VkBuffer accelBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory accelMemory_ = VK_NULL_HANDLE;
    VkBuffer readbackBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory readbackMemory_ = VK_NULL_HANDLE;

    VkDeviceSize particleBytes_ = 0;
    VkDeviceSize accelBytes_ = 0;
    VkDeviceSize accelReadbackOffset_ = 0;  // accel region start within readbackBuffer_

    VkDescriptorSetLayout setLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline forceCalcPipeline_ = VK_NULL_HANDLE;
    VkPipeline integratePipeline_ = VK_NULL_HANDLE;

    VkCommandPool commandPool_ = VK_NULL_HANDLE;
};

// Reference CPU implementation of the same O(N^2) force accumulation, in the
// same j = 0..N-1 iteration order as the shader.
std::vector<Vec3> cpuForceCalc(const std::vector<Particle>& particles,
                               const SimParams& params);

// Total linear momentum sum(m_i * v_i) of a particle set.
Vec3 totalMomentum(const std::vector<Particle>& particles);
