#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <vector>

#ifdef NDEBUG
static constexpr bool kEnableValidation = false;
#else
static constexpr bool kEnableValidation = true;
#endif

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT /*type*/,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void* /*userData*/) {
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        std::fprintf(stderr, "[validation] %s\n", data->pMessage);
    }
    return VK_FALSE;
}

static bool hasLayer(const char* name) {
    uint32_t count = 0;
    vkEnumerateInstanceLayerProperties(&count, nullptr);
    std::vector<VkLayerProperties> layers(count);
    vkEnumerateInstanceLayerProperties(&count, layers.data());
    for (const auto& l : layers) {
        if (std::strcmp(l.layerName, name) == 0) return true;
    }
    return false;
}

static bool hasInstanceExtension(const char* name) {
    uint32_t count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> exts(count);
    vkEnumerateInstanceExtensionProperties(nullptr, &count, exts.data());
    for (const auto& e : exts) {
        if (std::strcmp(e.extensionName, name) == 0) return true;
    }
    return false;
}

int main() {
    // --- Instance ---
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "gpu-compute-profiler";
    appInfo.apiVersion = VK_API_VERSION_1_2;

    std::vector<const char*> instanceExtensions;
    std::vector<const char*> instanceLayers;
    VkInstanceCreateFlags instanceFlags = 0;

    if (hasInstanceExtension(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
        instanceExtensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        instanceFlags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    }

    bool validationAvailable = kEnableValidation && hasLayer("VK_LAYER_KHRONOS_validation");
    if (validationAvailable) {
        instanceLayers.push_back("VK_LAYER_KHRONOS_validation");
        instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    } else if (kEnableValidation) {
        std::fprintf(stderr,
                     "warning: VK_LAYER_KHRONOS_validation not found, running without "
                     "validation\n");
    }

    VkInstanceCreateInfo instInfo{};
    instInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instInfo.pApplicationInfo = &appInfo;
    instInfo.flags = instanceFlags;
    instInfo.enabledExtensionCount = static_cast<uint32_t>(instanceExtensions.size());
    instInfo.ppEnabledExtensionNames = instanceExtensions.data();
    instInfo.enabledLayerCount = static_cast<uint32_t>(instanceLayers.size());
    instInfo.ppEnabledLayerNames = instanceLayers.data();

    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&instInfo, nullptr, &instance) != VK_SUCCESS) {
        std::fprintf(stderr, "fatal: vkCreateInstance failed\n");
        return EXIT_FAILURE;
    }

    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    if (validationAvailable) {
        auto createFn = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
        if (createFn) {
            VkDebugUtilsMessengerCreateInfoEXT dbgInfo{};
            dbgInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
            dbgInfo.messageSeverity =
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            dbgInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                   VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                   VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            dbgInfo.pfnUserCallback = debugCallback;
            createFn(instance, &dbgInfo, nullptr, &messenger);
        }
    }

    // --- Physical device: pick first with a compute-capable queue family ---
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        std::fprintf(stderr, "fatal: no Vulkan physical devices found\n");
        return EXIT_FAILURE;
    }
    std::vector<VkPhysicalDevice> physicalDevices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, physicalDevices.data());

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    uint32_t computeQueueFamily = 0;
    for (auto pd : physicalDevices) {
        uint32_t qCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qCount, nullptr);
        std::vector<VkQueueFamilyProperties> qProps(qCount);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qCount, qProps.data());
        for (uint32_t i = 0; i < qCount; ++i) {
            if (qProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                physicalDevice = pd;
                computeQueueFamily = i;
                break;
            }
        }
        if (physicalDevice != VK_NULL_HANDLE) break;
    }
    if (physicalDevice == VK_NULL_HANDLE) {
        std::fprintf(stderr, "fatal: no compute-capable queue family found\n");
        return EXIT_FAILURE;
    }

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(physicalDevice, &props);

    // --- Logical device + compute queue ---
    std::vector<const char*> deviceExtensions;
    uint32_t devExtCount = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &devExtCount, nullptr);
    std::vector<VkExtensionProperties> devExts(devExtCount);
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &devExtCount, devExts.data());
    for (const auto& e : devExts) {
        if (std::strcmp(e.extensionName, "VK_KHR_portability_subset") == 0) {
            deviceExtensions.push_back("VK_KHR_portability_subset");
        }
    }

    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = computeQueueFamily;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo devInfo{};
    devInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    devInfo.queueCreateInfoCount = 1;
    devInfo.pQueueCreateInfos = &queueInfo;
    devInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    devInfo.ppEnabledExtensionNames = deviceExtensions.data();

    VkDevice device = VK_NULL_HANDLE;
    if (vkCreateDevice(physicalDevice, &devInfo, nullptr, &device) != VK_SUCCESS) {
        std::fprintf(stderr, "fatal: vkCreateDevice failed\n");
        return EXIT_FAILURE;
    }

    VkQueue computeQueue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, computeQueueFamily, 0, &computeQueue);

    bool isTranslationLayer = std::strstr(props.deviceName, "MoltenVK") != nullptr ||
                               props.vendorID == 0x106b;  // Apple vendor ID via MoltenVK
    std::printf("Vulkan device: %s\n", props.deviceName);
    std::printf("Driver version: %u.%u.%u\n", VK_VERSION_MAJOR(props.driverVersion),
                VK_VERSION_MINOR(props.driverVersion), VK_VERSION_PATCH(props.driverVersion));
    std::printf("API version: %u.%u.%u\n", VK_VERSION_MAJOR(props.apiVersion),
                VK_VERSION_MINOR(props.apiVersion), VK_VERSION_PATCH(props.apiVersion));
    std::printf("Compute queue family: %u\n", computeQueueFamily);
    std::printf("Translation layer (MoltenVK/Metal, timings not native): %s\n",
                isTranslationLayer ? "yes" : "no");

    // --- Teardown ---
    vkDestroyDevice(device, nullptr);
    if (messenger != VK_NULL_HANDLE) {
        auto destroyFn = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
        if (destroyFn) destroyFn(instance, messenger, nullptr);
    }
    vkDestroyInstance(instance, nullptr);

    return EXIT_SUCCESS;
}
