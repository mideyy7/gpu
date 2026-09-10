#include "simulation.h"

#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#ifndef SHADER_DIR
#define SHADER_DIR "shaders"
#endif

namespace {

void check(VkResult r, const char* what) {
    if (r != VK_SUCCESS)
        throw std::runtime_error(std::string(what) + " failed (VkResult " +
                                 std::to_string(static_cast<int>(r)) + ")");
}

// Combined push-constant block shared by both compute pipelines.
struct PushConstants {
    uint32_t particleCount;
    float gravity;
    float softening;
    float dt;
};

VkDeviceSize alignUp(VkDeviceSize v, VkDeviceSize a) { return (v + a - 1) / a * a; }

}  // namespace

void Simulation::init(VulkanContext& ctx, const SimParams& params,
                      const std::vector<Particle>& initial) {
    ctx_ = &ctx;
    params_ = params;
    if (initial.size() != params.particleCount)
        throw std::runtime_error("initial particle count mismatch");

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = ctx.computeQueueFamily;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    check(vkCreateCommandPool(ctx.device, &poolInfo, nullptr, &commandPool_),
          "vkCreateCommandPool");

    createBuffers(initial);
    createPipelines();
}

void Simulation::createBuffers(const std::vector<Particle>& initial) {
    auto& ctx = *ctx_;
    particleBytes_ = sizeof(Particle) * params_.particleCount;
    accelBytes_ = sizeof(float) * 4 * params_.particleCount;

    auto makeBuffer = [&](VkDeviceSize size, VkBufferUsageFlags usage,
                          VkMemoryPropertyFlags memFlags, VkBuffer& buf,
                          VkDeviceMemory& mem) {
        VkBufferCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        info.size = size;
        info.usage = usage;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        check(vkCreateBuffer(ctx.device, &info, nullptr, &buf), "vkCreateBuffer");
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(ctx.device, buf, &req);
        VkMemoryAllocateInfo alloc{};
        alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc.allocationSize = req.size;
        alloc.memoryTypeIndex = ctx.findMemoryType(req.memoryTypeBits, memFlags);
        check(vkAllocateMemory(ctx.device, &alloc, nullptr, &mem), "vkAllocateMemory");
        check(vkBindBufferMemory(ctx.device, buf, mem, 0), "vkBindBufferMemory");
    };

    makeBuffer(particleBytes_,
               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, particleBuffer_, particleMemory_);
    makeBuffer(accelBytes_,
               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, accelBuffer_, accelMemory_);

    accelReadbackOffset_ = alignUp(particleBytes_, 256);
    makeBuffer(accelReadbackOffset_ + accelBytes_, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
               readbackBuffer_, readbackMemory_);

    // Upload initial particle state via a staging buffer.
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory stagingMem = VK_NULL_HANDLE;
    makeBuffer(particleBytes_, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
               staging, stagingMem);
    void* mapped = nullptr;
    check(vkMapMemory(ctx.device, stagingMem, 0, particleBytes_, 0, &mapped), "vkMapMemory");
    std::memcpy(mapped, initial.data(), static_cast<size_t>(particleBytes_));
    vkUnmapMemory(ctx.device, stagingMem);

    VkCommandBuffer cmd = beginCommands();
    VkBufferCopy region{0, 0, particleBytes_};
    vkCmdCopyBuffer(cmd, staging, particleBuffer_, 1, &region);
    submitAndWait(cmd);

    vkDestroyBuffer(ctx.device, staging, nullptr);
    vkFreeMemory(ctx.device, stagingMem, nullptr);
}

void Simulation::createPipelines() {
    auto& ctx = *ctx_;

    std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
    for (uint32_t i = 0; i < bindings.size(); ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    check(vkCreateDescriptorSetLayout(ctx.device, &layoutInfo, nullptr, &setLayout_),
          "vkCreateDescriptorSetLayout");

    VkPushConstantRange pcRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants)};
    VkPipelineLayoutCreateInfo plInfo{};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = 1;
    plInfo.pSetLayouts = &setLayout_;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges = &pcRange;
    check(vkCreatePipelineLayout(ctx.device, &plInfo, nullptr, &pipelineLayout_),
          "vkCreatePipelineLayout");

    auto makePipeline = [&](const char* file, VkPipeline& out) {
        VkShaderModule mod =
            loadShaderModule(ctx.device, std::string(SHADER_DIR) + "/" + file);
        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = mod;
        stage.pName = "main";
        VkComputePipelineCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        info.stage = stage;
        info.layout = pipelineLayout_;
        check(vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &info, nullptr, &out),
              "vkCreateComputePipelines");
        vkDestroyShaderModule(ctx.device, mod, nullptr);
    };
    makePipeline("force_calc.comp.spv", forceCalcPipeline_);
    makePipeline("integrate.comp.spv", integratePipeline_);

    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    check(vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &descriptorPool_),
          "vkCreateDescriptorPool");

    VkDescriptorSetAllocateInfo dsAlloc{};
    dsAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsAlloc.descriptorPool = descriptorPool_;
    dsAlloc.descriptorSetCount = 1;
    dsAlloc.pSetLayouts = &setLayout_;
    check(vkAllocateDescriptorSets(ctx.device, &dsAlloc, &descriptorSet_),
          "vkAllocateDescriptorSets");

    VkDescriptorBufferInfo particleInfo{particleBuffer_, 0, particleBytes_};
    VkDescriptorBufferInfo accelInfo{accelBuffer_, 0, accelBytes_};
    std::array<VkWriteDescriptorSet, 2> writes{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = descriptorSet_;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo = &particleInfo;
    writes[1] = writes[0];
    writes[1].dstBinding = 1;
    writes[1].pBufferInfo = &accelInfo;
    vkUpdateDescriptorSets(ctx.device, static_cast<uint32_t>(writes.size()), writes.data(), 0,
                           nullptr);
}

VkCommandBuffer Simulation::beginCommands() {
    auto& ctx = *ctx_;
    VkCommandBufferAllocateInfo cbAlloc{};
    cbAlloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbAlloc.commandPool = commandPool_;
    cbAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbAlloc.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    check(vkAllocateCommandBuffers(ctx.device, &cbAlloc, &cmd), "vkAllocateCommandBuffers");
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(cmd, &begin), "vkBeginCommandBuffer");
    return cmd;
}

void Simulation::submitAndWait(VkCommandBuffer cmd) {
    auto& ctx = *ctx_;
    check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    check(vkCreateFence(ctx.device, &fenceInfo, nullptr, &fence), "vkCreateFence");
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    check(vkQueueSubmit(ctx.computeQueue, 1, &submit, fence), "vkQueueSubmit");
    check(vkWaitForFences(ctx.device, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
    vkDestroyFence(ctx.device, fence, nullptr);
    vkFreeCommandBuffers(ctx.device, commandPool_, 1, &cmd);
}

void Simulation::runForceCalcOnce() {
    VkCommandBuffer cmd = beginCommands();
    PushConstants pc{params_.particleCount, params_.gravity, params_.softening, params_.dt};

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, forceCalcPipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_, 0, 1,
                            &descriptorSet_, 0, nullptr);
    vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    uint32_t groups = (params_.particleCount + 63) / 64;
    vkCmdDispatch(cmd, groups, 1, 1);

    VkMemoryBarrier toTransfer{};
    toTransfer.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    toTransfer.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &toTransfer, 0, nullptr, 0,
                         nullptr);

    VkBufferCopy region{0, accelReadbackOffset_, accelBytes_};
    vkCmdCopyBuffer(cmd, accelBuffer_, readbackBuffer_, 1, &region);

    submitAndWait(cmd);
}

void Simulation::stepOnce() {
    VkCommandBuffer cmd = beginCommands();
    PushConstants pc{params_.particleCount, params_.gravity, params_.softening, params_.dt};
    uint32_t groups = (params_.particleCount + 63) / 64;

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_, 0, 1,
                            &descriptorSet_, 0, nullptr);
    vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, forceCalcPipeline_);
    vkCmdDispatch(cmd, groups, 1, 1);

    // accel: shader write -> shader read hazard before integrate consumes it.
    VkMemoryBarrier computeToCompute{};
    computeToCompute.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    computeToCompute.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    computeToCompute.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &computeToCompute, 0,
                         nullptr, 0, nullptr);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, integratePipeline_);
    vkCmdDispatch(cmd, groups, 1, 1);

    // particles: shader write -> transfer read before the readback copy.
    VkMemoryBarrier computeToTransfer{};
    computeToTransfer.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    computeToTransfer.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    computeToTransfer.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &computeToTransfer, 0, nullptr,
                         0, nullptr);

    VkBufferCopy region{0, 0, particleBytes_};
    vkCmdCopyBuffer(cmd, particleBuffer_, readbackBuffer_, 1, &region);

    submitAndWait(cmd);
}

std::vector<Vec3> Simulation::readAccelerations() {
    auto& ctx = *ctx_;
    void* mapped = nullptr;
    check(vkMapMemory(ctx.device, readbackMemory_, accelReadbackOffset_, accelBytes_, 0,
                      &mapped),
          "vkMapMemory");
    const float* f = static_cast<const float*>(mapped);
    std::vector<Vec3> out(params_.particleCount);
    for (uint32_t i = 0; i < params_.particleCount; ++i)
        out[i] = {f[i * 4 + 0], f[i * 4 + 1], f[i * 4 + 2]};
    vkUnmapMemory(ctx.device, readbackMemory_);
    return out;
}

std::vector<Particle> Simulation::readParticles() {
    auto& ctx = *ctx_;
    void* mapped = nullptr;
    check(vkMapMemory(ctx.device, readbackMemory_, 0, particleBytes_, 0, &mapped),
          "vkMapMemory");
    std::vector<Particle> out(params_.particleCount);
    std::memcpy(out.data(), mapped, static_cast<size_t>(particleBytes_));
    vkUnmapMemory(ctx.device, readbackMemory_);
    return out;
}

void Simulation::destroy() {
    if (!ctx_) return;
    VkDevice d = ctx_->device;
    vkDeviceWaitIdle(d);
    if (forceCalcPipeline_) vkDestroyPipeline(d, forceCalcPipeline_, nullptr);
    if (integratePipeline_) vkDestroyPipeline(d, integratePipeline_, nullptr);
    if (pipelineLayout_) vkDestroyPipelineLayout(d, pipelineLayout_, nullptr);
    if (descriptorPool_) vkDestroyDescriptorPool(d, descriptorPool_, nullptr);
    if (setLayout_) vkDestroyDescriptorSetLayout(d, setLayout_, nullptr);
    if (particleBuffer_) vkDestroyBuffer(d, particleBuffer_, nullptr);
    if (particleMemory_) vkFreeMemory(d, particleMemory_, nullptr);
    if (accelBuffer_) vkDestroyBuffer(d, accelBuffer_, nullptr);
    if (accelMemory_) vkFreeMemory(d, accelMemory_, nullptr);
    if (readbackBuffer_) vkDestroyBuffer(d, readbackBuffer_, nullptr);
    if (readbackMemory_) vkFreeMemory(d, readbackMemory_, nullptr);
    if (commandPool_) vkDestroyCommandPool(d, commandPool_, nullptr);
    ctx_ = nullptr;
}

std::vector<Vec3> cpuForceCalc(const std::vector<Particle>& particles,
                               const SimParams& params) {
    std::vector<Vec3> accel(particles.size());
    const float soft2 = params.softening * params.softening;
    for (size_t i = 0; i < particles.size(); ++i) {
        float ax = 0, ay = 0, az = 0;
        const float pix = particles[i].px, piy = particles[i].py, piz = particles[i].pz;
        for (size_t j = 0; j < particles.size(); ++j) {
            if (j == i) continue;
            const float dx = particles[j].px - pix;
            const float dy = particles[j].py - piy;
            const float dz = particles[j].pz - piz;
            const float distSqr = dx * dx + dy * dy + dz * dz + soft2;
            const float invDist = 1.0f / std::sqrt(distSqr);
            const float invDist3 = invDist * invDist * invDist;
            const float s = params.gravity * particles[j].mass * invDist3;
            ax += s * dx;
            ay += s * dy;
            az += s * dz;
        }
        accel[i] = {ax, ay, az};
    }
    return accel;
}

Vec3 totalMomentum(const std::vector<Particle>& particles) {
    double mx = 0, my = 0, mz = 0;
    for (const auto& p : particles) {
        mx += static_cast<double>(p.mass) * p.vx;
        my += static_cast<double>(p.mass) * p.vy;
        mz += static_cast<double>(p.mass) * p.vz;
    }
    return {static_cast<float>(mx), static_cast<float>(my), static_cast<float>(mz)};
}
