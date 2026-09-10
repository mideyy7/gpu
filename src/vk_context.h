#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>
#include <vector>

// Minimal headless compute Vulkan context: instance (+ optional validation),
// a compute-capable physical device, logical device and one compute queue.
struct VulkanContext {
    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties props{};
    VkPhysicalDeviceMemoryProperties memProps{};
    VkDevice device = VK_NULL_HANDLE;
    uint32_t computeQueueFamily = 0;
    VkQueue computeQueue = VK_NULL_HANDLE;

    bool validationEnabled = false;
    bool isTranslationLayer = false;

    void create(bool enableValidation);
    void destroy();

    // Picks a memory type index satisfying typeFilter and all required flags.
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags required) const;

    std::string deviceSummary() const;
};

// Load a SPIR-V file and wrap it in a VkShaderModule.
VkShaderModule loadShaderModule(VkDevice device, const std::string& path);
