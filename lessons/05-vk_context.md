# Part 05 — `vk_context.{h,cpp}`: Vulkan start-up

Files: `src/vk_context.h` (34 lines), `src/vk_context.cpp` (204 lines).

**Job of this file:** do the one-time Vulkan bring-up so the rest of the
program has a working `VkDevice` and a compute `VkQueue`. It also provides
two helpers everyone reuses: `findMemoryType` and `loadShaderModule`.

Recall the order from Part 02: instance → physical device → queue family →
logical device → queue. This file is that sequence, plus optional validation
wiring.

---

## 1. The header, `vk_context.h`

```cpp
#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <string>
#include <vector>
```

Include guard, the Vulkan API, and the three std pieces the header's
signatures mention.

```cpp
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
```

A plain data bundle (it's a `struct`, all public). Every handle starts at
`VK_NULL_HANDLE` (Vulkan's "null pointer"), the two property structs start
zeroed (`{}`), the flags start `false`. So a freshly declared
`VulkanContext` is in a well-defined "nothing created yet" state.

- `props` — device name, limits (including `timestampPeriod`), vendor ID.
- `memProps` — the list of memory types, used by `findMemoryType`.
- `computeQueueFamily` — index of the queue family we'll use, needed later
  for command pools and timestamp support checks.
- `isTranslationLayer` — set true if this is MoltenVK; drives the "timings
  not native" warning.

```cpp
    void create(bool enableValidation);
    void destroy();
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags required) const;
    std::string deviceSummary() const;
};

VkShaderModule loadShaderModule(VkDevice device, const std::string& path);
```

Four methods and one free function (declared here, defined in the `.cpp`).
`const` on the last two = "doesn't modify the context." `loadShaderModule`
is a free function because it doesn't need a `VulkanContext` — just a
`VkDevice` and a path.

---

## 2. The anonymous namespace helpers

```cpp
namespace {
```

Everything until the matching `}` has **internal linkage** — it's private to
this `.cpp` file. Another file could define its own `check` with no clash.
This is the C++ replacement for file-scope `static` functions.

### `debugCallback`

```cpp
VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT /*type*/,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void* /*userData*/) {
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        std::fprintf(stderr, "[validation] %s\n", data->pMessage);
    }
    return VK_FALSE;
}
```

This is the function the **validation layer** calls whenever it has
something to say. Vulkan calls it, so it must have the exact signature
Vulkan expects (`VKAPI_ATTR`/`VKAPI_CALL` fix the calling convention;
`VkBool32` is Vulkan's 32-bit bool).

- `/*type*/` and `/*userData*/` — the parameter is unused, so its name is
  commented out to silence "unused parameter" warnings while keeping the
  signature correct.
- `severity >= ..._WARNING_BIT_EXT` — only print warnings and errors, not
  chatty info messages. The severities are ordered enum values, so `>=`
  works.
- `data->pMessage` — the human-readable string.
- `return VK_FALSE` — "do not abort the call that triggered this." (Returning
  `VK_TRUE` would make the offending Vulkan call fail on the spot; that's
  for aggressive debugging.)

**Why it matters:** every "0 validation errors" claim in the project rests
on this callback. If validation had anything to complain about, it would
land on `stderr` with a `[validation]` prefix and the test scripts count
`VUID` occurrences.

### `hasLayer` and `hasInstanceExtension`

```cpp
bool hasLayer(const char* name) {
    uint32_t count = 0;
    vkEnumerateInstanceLayerProperties(&count, nullptr);
    std::vector<VkLayerProperties> layers(count);
    vkEnumerateInstanceLayerProperties(&count, layers.data());
    for (const auto& l : layers)
        if (std::strcmp(l.layerName, name) == 0) return true;
    return false;
}
```

The **count-then-data** pattern (Part 02 §7): first call with `nullptr`
gets the count, allocate a vector, second call fills it. Then linear-scan
for a layer whose `layerName` C-string equals `name` (`strcmp(...) == 0`).

`hasInstanceExtension` is identical but for extensions
(`vkEnumerateInstanceExtensionProperties`, first arg `nullptr` = "not scoped
to a specific layer").

Used to make start-up *robust*: only request the validation layer if it's
actually installed; only request the portability extension if it exists.

### `check`

```cpp
void check(VkResult r, const char* what) {
    if (r != VK_SUCCESS) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "%s failed (VkResult %d)", what, static_cast<int>(r));
        throw std::runtime_error(buf);
    }
}
```

Almost every Vulkan function returns a `VkResult` (an enum; `VK_SUCCESS` =
0). `check` turns "not success" into an exception carrying a message like
`"vkCreateInstance failed (VkResult -3)"`. `main` catches it and prints
`fatal: ...`. This keeps the happy-path code readable — one `check(...)`
wrapper instead of an `if` after every call.

(`simulation.cpp` has its own near-identical `check` in its own anonymous
namespace — same idea, kept local.)

---

## 3. `VulkanContext::create` — the main sequence

### Instance creation

```cpp
VkApplicationInfo appInfo{};
appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
appInfo.pApplicationName = "gpu-compute-profiler";
appInfo.apiVersion = VK_API_VERSION_1_2;
```

Optional metadata the driver may use. `apiVersion` = "I'm written against
Vulkan 1.2."

```cpp
std::vector<const char*> extensions;
std::vector<const char*> layers;
VkInstanceCreateFlags flags = 0;
```

We'll build up the lists of extension names, layer names, and flag bits to
request.

```cpp
if (hasInstanceExtension(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
    extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
}
```

**Portability** = MoltenVK-style non-conformant implementations. On macOS you
*must* opt in to "enumerate portability devices" or `vkEnumeratePhysicalDevices`
returns nothing. On Linux this extension doesn't exist, `hasInstanceExtension`
returns false, and the block is skipped. This is what makes the same code
build and run on both.

```cpp
validationEnabled = enableValidation && hasLayer("VK_LAYER_KHRONOS_validation");
if (validationEnabled) {
    layers.push_back("VK_LAYER_KHRONOS_validation");
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
} else if (enableValidation) {
    std::fprintf(stderr, "warning: validation layer not found, continuing without it\n");
}
```

Enable validation only if the caller asked *and* the layer is present. If
enabled, also enable `VK_EXT_debug_utils` (the extension that lets us
register `debugCallback`). If asked-but-missing, warn and carry on
un-validated (this is what you saw when the layer path wasn't set up).

```cpp
VkInstanceCreateInfo instInfo{};
instInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
instInfo.pApplicationInfo = &appInfo;
instInfo.flags = flags;
instInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
instInfo.ppEnabledExtensionNames = extensions.data();
instInfo.enabledLayerCount = static_cast<uint32_t>(layers.size());
instInfo.ppEnabledLayerNames = layers.data();

check(vkCreateInstance(&instInfo, nullptr, &instance), "vkCreateInstance");
```

Fill the create-info: count + pointer for each list (the classic C way to
pass an array). `nullptr` for the allocator. `&instance` receives the
handle. `check` throws on failure.

### Debug messenger

```cpp
if (validationEnabled) {
    auto createFn = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
    if (createFn) {
        VkDebugUtilsMessengerCreateInfoEXT dbg{};
        dbg.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        dbg.messageSeverity = ..._WARNING_BIT_EXT | ..._ERROR_BIT_EXT;
        dbg.messageType     = ..._GENERAL | ..._VALIDATION | ..._PERFORMANCE;
        dbg.pfnUserCallback = debugCallback;
        createFn(instance, &dbg, nullptr, &messenger);
    }
}
```

`vkCreateDebugUtilsMessengerEXT` comes from an **extension**, so it's not a
normal linkable symbol — you fetch its address at runtime with
`vkGetInstanceProcAddr(instance, "name")`, which returns a generic function
pointer, then `reinterpret_cast` it to the correct typed pointer
`PFN_vkCreateDebugUtilsMessengerEXT`. If the fetch returns null (extension
not really there), skip.

The create-info says "call me for warnings and errors, of all three
categories, at `debugCallback`." `messenger` is stored so `destroy` can
tear it down.

### Physical device selection

```cpp
uint32_t deviceCount = 0;
vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
if (deviceCount == 0) throw std::runtime_error("no Vulkan physical devices");
std::vector<VkPhysicalDevice> devices(deviceCount);
vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
```

Count-then-data again. No GPU at all → fatal.

```cpp
for (auto pd : devices) {
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
if (physicalDevice == VK_NULL_HANDLE)
    throw std::runtime_error("no compute-capable queue family");
```

For each GPU, list its queue families; for each family, test the
**compute** bit with `flags & VK_QUEUE_COMPUTE_BIT`. Take the **first** GPU
that has a compute family, remember `(physicalDevice, computeQueueFamily)`,
and stop. Simple policy — fine because this project has modest needs. (A
production tool might prefer a discrete GPU, a dedicated compute queue,
etc.)

```cpp
vkGetPhysicalDeviceProperties(physicalDevice, &props);
vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);
isTranslationLayer =
    std::strstr(props.deviceName, "MoltenVK") != nullptr || props.vendorID == 0x106b;
```

Cache the properties and memory properties (used all over later).
`strstr(a, b)` returns a pointer to the first occurrence of `b` in `a`, or
`nullptr`. So this is "device name contains 'MoltenVK', OR vendor ID is
Apple's (`0x106b`)" → mark it a translation layer.

### Logical device creation

```cpp
std::vector<const char*> deviceExtensions;
uint32_t devExtCount = 0;
vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &devExtCount, nullptr);
std::vector<VkExtensionProperties> devExts(devExtCount);
vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &devExtCount, devExts.data());
for (const auto& e : devExts)
    if (std::strcmp(e.extensionName, "VK_KHR_portability_subset") == 0)
        deviceExtensions.push_back("VK_KHR_portability_subset");
```

On MoltenVK the device advertises `VK_KHR_portability_subset`, and Vulkan
**requires** you to enable it if it's present. So: if it's in the list, add
it to the list we'll request. On a native driver it's absent and nothing
happens.

```cpp
float priority = 1.0f;
VkDeviceQueueCreateInfo qInfo{};
qInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
qInfo.queueFamilyIndex = computeQueueFamily;
qInfo.queueCount = 1;
qInfo.pQueuePriorities = &priority;
```

"From family `computeQueueFamily`, give me **1** queue, priority 1.0."
`pQueuePriorities` must point to an array of `queueCount` floats in `[0,1]`.

```cpp
VkDeviceCreateInfo devInfo{};
devInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
devInfo.queueCreateInfoCount = 1;
devInfo.pQueueCreateInfos = &qInfo;
devInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
devInfo.ppEnabledExtensionNames = deviceExtensions.data();

check(vkCreateDevice(physicalDevice, &devInfo, nullptr, &device), "vkCreateDevice");
vkGetDeviceQueue(device, computeQueueFamily, 0, &computeQueue);
```

Create the logical `device` (with the one queue and any required
extensions), then fetch queue index 0 of that family into `computeQueue`.
After this line, the context is fully usable.

---

## 4. `VulkanContext::destroy`

```cpp
void VulkanContext::destroy() {
    if (device != VK_NULL_HANDLE) vkDestroyDevice(device, nullptr);
    if (messenger != VK_NULL_HANDLE) {
        auto destroyFn = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
        if (destroyFn) destroyFn(instance, messenger, nullptr);
    }
    if (instance != VK_NULL_HANDLE) vkDestroyInstance(instance, nullptr);
    *this = VulkanContext{};
}
```

Tear down in **reverse creation order**: device, then messenger (again an
extension function fetched by name), then instance. Each guarded by
`if (handle)`. `*this = VulkanContext{}` resets every member to default so
the object is safely reusable / a second `destroy()` is a no-op.

(Note the ordering nuance: `vkDestroyDevice` is called before the messenger,
which is fine because the messenger belongs to the *instance*, not the
device.)

---

## 5. `findMemoryType`

```cpp
uint32_t VulkanContext::findMemoryType(uint32_t typeFilter,
                                       VkMemoryPropertyFlags required) const {
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((typeFilter & (1u << i)) &&
            (memProps.memoryTypes[i].propertyFlags & required) == required) {
            return i;
        }
    }
    throw std::runtime_error("no suitable memory type");
}
```

When you create a buffer, Vulkan tells you (via `VkMemoryRequirements`)
*which* memory types are acceptable for it — as a bitmask, `memoryTypeBits`.
That's `typeFilter` here.

You also want certain *properties* (e.g. `DEVICE_LOCAL`, or
`HOST_VISIBLE | HOST_COHERENT`). That's `required`.

The loop checks each memory type index `i`:

- `typeFilter & (1u << i)` — "is bit `i` set in the filter?" i.e. is type
  `i` acceptable for this buffer? (`1u << i` is "1 shifted left `i`
  places" = a mask with only bit `i` set.)
- `(propertyFlags & required) == required` — "does type `i` have **all** the
  flags I need?" (ANDing with `required` and comparing back to `required` is
  the idiom for "contains all these bits.")

Return the first index that satisfies both; throw if none do.

`simulation.cpp` calls this from `makeBuffer` for every buffer it creates.

---

## 6. `deviceSummary`

```cpp
std::string VulkanContext::deviceSummary() const {
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s (driver %u.%u.%u, API %u.%u.%u)%s", props.deviceName,
                  VK_VERSION_MAJOR(props.driverVersion), VK_VERSION_MINOR(...), VK_VERSION_PATCH(...),
                  VK_VERSION_MAJOR(props.apiVersion),    VK_VERSION_MINOR(...), VK_VERSION_PATCH(...),
                  isTranslationLayer ? " [translation layer: timings not native]" : "");
    return buf;
}
```

Formats a one-line string like
`Apple M1 (driver 0.2.2210, API 1.2.357) [translation layer: timings not native]`.
`VK_VERSION_MAJOR/MINOR/PATCH` are macros that unpack a packed 32-bit
version integer. The ternary appends the caveat only when it applies.
`return buf;` — the `char[256]` is implicitly converted to a `std::string`
(which copies it), so returning a local buffer is safe here.

This string goes into the report header and the `Device:` line at start-up,
so the MoltenVK caveat is always visible next to the numbers.

---

## 7. `loadShaderModule`

```cpp
VkShaderModule loadShaderModule(VkDevice device, const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("cannot open shader: " + path);
    size_t size = static_cast<size_t>(file.tellg());
    std::vector<char> code(size);
    file.seekg(0);
    file.read(code.data(), static_cast<std::streamsize>(size));

    VkShaderModuleCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = size;                                    // in BYTES
    info.pCode = reinterpret_cast<const uint32_t*>(code.data());  // as 32-bit words

    VkShaderModule mod = VK_NULL_HANDLE;
    check(vkCreateShaderModule(device, &info, nullptr, &mod), "vkCreateShaderModule");
    return mod;
}
```

- `std::ifstream file(path, binary | ate)` — open the `.spv` file in
  **binary** mode (don't translate newlines), positioned **at the end**
  (`ate`) so `tellg()` immediately returns the file size.
- `if (!file)` — `ifstream` converts to `false` if the open failed.
- Read `size` bytes into a `std::vector<char>`.
- SPIR-V is a stream of 32-bit words. `codeSize` is bytes, but `pCode` must
  be a `const uint32_t*`, hence the `reinterpret_cast` — "treat these bytes
  as an array of 32-bit words." (Safe: `std::vector`'s storage is suitably
  aligned, and `.spv` files are always a multiple of 4 bytes.)
- `vkCreateShaderModule` hands the SPIR-V to the driver, which may compile
  it further to native GPU code. Return the handle.

`simulation.cpp`'s `createPipelines` calls this twice, then destroys the
modules right after building the pipelines (the pipeline keeps whatever it
needs).

---

## Explain it in one sentence

> `vk_context` does the one-time Vulkan bring-up — instance (with validation
> and portability handled conditionally so the same code runs on MoltenVK
> and native Linux), pick the first compute-capable physical device, create
> a logical device with one compute queue — and exposes `findMemoryType`
> (choose a memory type matching a buffer's constraints and desired
> properties) and `loadShaderModule` (read a `.spv` file into a
> `VkShaderModule`).

## Questions you should be able to answer

1. Walk through `create()` in order: what Vulkan objects get made and why in
   that sequence?
2. Why are `hasLayer` / `hasInstanceExtension` used instead of just
   requesting things unconditionally?
3. How does the same code work on both MoltenVK (macOS) and a native Linux
   driver? Name the two conditional blocks.
4. Why is `vkCreateDebugUtilsMessengerEXT` fetched with
   `vkGetInstanceProcAddr` instead of called directly?
5. Explain the two bitmask tests inside `findMemoryType`.
6. Why does `loadShaderModule` open the file with `binary | ate`, and why
   the `reinterpret_cast<const uint32_t*>`?
7. What is `debugCallback` and why does "0 validation errors" depend on it?
