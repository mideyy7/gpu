// VK_LAYER_gcp_profiler -- a minimal reusable profiling layer.
//
// For any Vulkan app it counts submits and the compute-relevant recorded
// commands (dispatch, pipeline barrier, buffer copy) and times the CPU
// spent inside vkQueueSubmit and inside the fence / idle waits. On
// vkDestroyDevice it prints an aggregated report to stderr.
//
// This is the app-agnostic half of the in-app instrumentation: it
// reproduces the "submit call (CPU)" and "fence wait (CPU)" rows and the
// command structure for a binary it was never compiled into. It does not
// insert GPU timestamps into someone else's command buffers, so the
// per-named-stage GPU breakdown remains the in-app path's job.

#include <vulkan/vk_layer.h>
#include <vulkan/vulkan.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <unordered_map>

#if defined(__GNUC__)
#define GCP_EXPORT __attribute__((visibility("default")))
#else
#define GCP_EXPORT
#endif

namespace {

using Clock = std::chrono::steady_clock;
double nsBetween(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::nano>(b - a).count();
}

struct InstanceData {
    PFN_vkGetInstanceProcAddr gipa = nullptr;
    PFN_vkDestroyInstance DestroyInstance = nullptr;
};

struct DeviceData {
    PFN_vkGetDeviceProcAddr gdpa = nullptr;
    PFN_vkDestroyDevice DestroyDevice = nullptr;
    PFN_vkQueueSubmit QueueSubmit = nullptr;
    PFN_vkQueueWaitIdle QueueWaitIdle = nullptr;
    PFN_vkDeviceWaitIdle DeviceWaitIdle = nullptr;
    PFN_vkWaitForFences WaitForFences = nullptr;
    PFN_vkCmdDispatch CmdDispatch = nullptr;
    PFN_vkCmdPipelineBarrier CmdPipelineBarrier = nullptr;
    PFN_vkCmdCopyBuffer CmdCopyBuffer = nullptr;

    std::atomic<uint64_t> submits{0};
    std::atomic<uint64_t> dispatches{0};
    std::atomic<uint64_t> barriers{0};
    std::atomic<uint64_t> copies{0};
    std::atomic<uint64_t> waits{0};
    std::mutex mtx;
    double cpuSubmitNs = 0;
    double cpuWaitNs = 0;
};

std::mutex g_mapMtx;
std::unordered_map<void*, InstanceData*> g_inst;
std::unordered_map<void*, DeviceData*> g_dev;

void* dispatchKey(void* h) { return *reinterpret_cast<void**>(h); }

InstanceData* instData(void* h) {
    std::lock_guard<std::mutex> lk(g_mapMtx);
    return g_inst.at(dispatchKey(h));
}
DeviceData* devData(void* h) {
    std::lock_guard<std::mutex> lk(g_mapMtx);
    return g_dev.at(dispatchKey(h));
}

const VkLayerInstanceCreateInfo* findInstLink(const VkInstanceCreateInfo* ci) {
    auto* p = static_cast<const VkLayerInstanceCreateInfo*>(ci->pNext);
    while (p && !(p->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO &&
                  p->function == VK_LAYER_LINK_INFO))
        p = static_cast<const VkLayerInstanceCreateInfo*>(p->pNext);
    return p;
}
const VkLayerDeviceCreateInfo* findDevLink(const VkDeviceCreateInfo* ci) {
    auto* p = static_cast<const VkLayerDeviceCreateInfo*>(ci->pNext);
    while (p && !(p->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO &&
                  p->function == VK_LAYER_LINK_INFO))
        p = static_cast<const VkLayerDeviceCreateInfo*>(p->pNext);
    return p;
}

void printReport(DeviceData* d) {
    std::lock_guard<std::mutex> lk(d->mtx);
    const uint64_t s = d->submits, w = d->waits;
    std::fprintf(stderr,
                 "\n=== VK_LAYER_gcp_profiler report ===\n"
                 "  submits .............. %llu\n"
                 "  vkCmdDispatch ....... %llu   (%.2f / submit)\n"
                 "  vkCmdPipelineBarrier  %llu   (%.2f / submit)\n"
                 "  vkCmdCopyBuffer ..... %llu\n"
                 "  fence / idle waits .. %llu\n"
                 "  CPU in vkQueueSubmit  %.1f us total   %.1f us / submit\n"
                 "  CPU in waits ........ %.1f us total   %.1f us / wait\n"
                 "====================================\n",
                 (unsigned long long)s, (unsigned long long)d->dispatches,
                 s ? double(d->dispatches) / s : 0.0, (unsigned long long)d->barriers,
                 s ? double(d->barriers) / s : 0.0, (unsigned long long)d->copies,
                 (unsigned long long)w, d->cpuSubmitNs / 1e3,
                 s ? d->cpuSubmitNs / 1e3 / s : 0.0, d->cpuWaitNs / 1e3,
                 w ? d->cpuWaitNs / 1e3 / w : 0.0);
}

VKAPI_ATTR void VKAPI_CALL CmdDispatch(VkCommandBuffer cb, uint32_t x, uint32_t y,
                                      uint32_t z) {
    DeviceData* d = devData(cb);
    d->dispatches.fetch_add(1, std::memory_order_relaxed);
    d->CmdDispatch(cb, x, y, z);
}
VKAPI_ATTR void VKAPI_CALL CmdPipelineBarrier(
    VkCommandBuffer cb, VkPipelineStageFlags s, VkPipelineStageFlags dst,
    VkDependencyFlags df, uint32_t a, const VkMemoryBarrier* ap, uint32_t b,
    const VkBufferMemoryBarrier* bp, uint32_t c, const VkImageMemoryBarrier* cp) {
    DeviceData* d = devData(cb);
    d->barriers.fetch_add(1, std::memory_order_relaxed);
    d->CmdPipelineBarrier(cb, s, dst, df, a, ap, b, bp, c, cp);
}
VKAPI_ATTR void VKAPI_CALL CmdCopyBuffer(VkCommandBuffer cb, VkBuffer s, VkBuffer dst,
                                        uint32_t rc, const VkBufferCopy* r) {
    DeviceData* d = devData(cb);
    d->copies.fetch_add(1, std::memory_order_relaxed);
    d->CmdCopyBuffer(cb, s, dst, rc, r);
}

VKAPI_ATTR VkResult VKAPI_CALL QueueSubmit(VkQueue q, uint32_t c,
                                          const VkSubmitInfo* s, VkFence f) {
    DeviceData* d = devData(q);
    d->submits.fetch_add(c, std::memory_order_relaxed);
    const Clock::time_point t0 = Clock::now();
    VkResult r = d->QueueSubmit(q, c, s, f);
    const double dt = nsBetween(t0, Clock::now());
    std::lock_guard<std::mutex> lk(d->mtx);
    d->cpuSubmitNs += dt;
    return r;
}

template <class Fn>
VkResult timedWait(DeviceData* d, Fn&& fn) {
    d->waits.fetch_add(1, std::memory_order_relaxed);
    const Clock::time_point t0 = Clock::now();
    VkResult r = fn();
    const double dt = nsBetween(t0, Clock::now());
    std::lock_guard<std::mutex> lk(d->mtx);
    d->cpuWaitNs += dt;
    return r;
}

VKAPI_ATTR VkResult VKAPI_CALL WaitForFences(VkDevice dev, uint32_t c,
                                            const VkFence* fences, VkBool32 all,
                                            uint64_t to) {
    DeviceData* d = devData(dev);
    return timedWait(d, [&] { return d->WaitForFences(dev, c, fences, all, to); });
}
VKAPI_ATTR VkResult VKAPI_CALL QueueWaitIdle(VkQueue q) {
    DeviceData* d = devData(q);
    return timedWait(d, [&] { return d->QueueWaitIdle(q); });
}
VKAPI_ATTR VkResult VKAPI_CALL DeviceWaitIdle(VkDevice dev) {
    DeviceData* d = devData(dev);
    return timedWait(d, [&] { return d->DeviceWaitIdle(dev); });
}

VKAPI_ATTR VkResult VKAPI_CALL CreateInstance(const VkInstanceCreateInfo* ci,
                                             const VkAllocationCallbacks* alloc,
                                             VkInstance* out) {
    auto* link = findInstLink(ci);
    auto gipa = link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    const_cast<VkLayerInstanceCreateInfo*>(link)->u.pLayerInfo =
        link->u.pLayerInfo->pNext;

    auto createInstance = (PFN_vkCreateInstance)gipa(VK_NULL_HANDLE, "vkCreateInstance");
    VkResult r = createInstance(ci, alloc, out);
    if (r != VK_SUCCESS) return r;

    auto* id = new InstanceData();
    id->gipa = gipa;
    id->DestroyInstance = (PFN_vkDestroyInstance)gipa(*out, "vkDestroyInstance");
    std::lock_guard<std::mutex> lk(g_mapMtx);
    g_inst[dispatchKey(*out)] = id;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL DestroyInstance(VkInstance inst,
                                          const VkAllocationCallbacks* alloc) {
    InstanceData* id = instData(inst);
    auto fn = id->DestroyInstance;
    {
        std::lock_guard<std::mutex> lk(g_mapMtx);
        g_inst.erase(dispatchKey(inst));
    }
    delete id;
    fn(inst, alloc);
}

VKAPI_ATTR VkResult VKAPI_CALL CreateDevice(VkPhysicalDevice phys,
                                            const VkDeviceCreateInfo* ci,
                                            const VkAllocationCallbacks* alloc,
                                            VkDevice* out) {
    auto* link = findDevLink(ci);
    auto gipa = link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    auto gdpa = link->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    const_cast<VkLayerDeviceCreateInfo*>(link)->u.pLayerInfo = link->u.pLayerInfo->pNext;

    auto createDevice = (PFN_vkCreateDevice)gipa(VK_NULL_HANDLE, "vkCreateDevice");
    VkResult r = createDevice(phys, ci, alloc, out);
    if (r != VK_SUCCESS) return r;

    auto* dd = new DeviceData();
    dd->gdpa = gdpa;
#define H(name) dd->name = (PFN_vk##name)gdpa(*out, "vk" #name)
    H(DestroyDevice);
    H(QueueSubmit);
    H(QueueWaitIdle);
    H(DeviceWaitIdle);
    H(WaitForFences);
    H(CmdDispatch);
    H(CmdPipelineBarrier);
    H(CmdCopyBuffer);
#undef H
    std::lock_guard<std::mutex> lk(g_mapMtx);
    g_dev[dispatchKey(*out)] = dd;
    std::fprintf(stderr, "[VK_LAYER_gcp_profiler] attached to new device\n");
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL DestroyDevice(VkDevice dev,
                                        const VkAllocationCallbacks* alloc) {
    DeviceData* dd = devData(dev);
    printReport(dd);
    auto fn = dd->DestroyDevice;
    {
        std::lock_guard<std::mutex> lk(g_mapMtx);
        g_dev.erase(dispatchKey(dev));
    }
    delete dd;
    fn(dev, alloc);
}

#define RET(name) \
    if (!std::strcmp(p, "vk" #name)) return reinterpret_cast<PFN_vkVoidFunction>(name)

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetDeviceProcAddr(VkDevice dev, const char* p) {
    RET(GetDeviceProcAddr);
    RET(DestroyDevice);
    RET(QueueSubmit);
    RET(QueueWaitIdle);
    RET(DeviceWaitIdle);
    RET(WaitForFences);
    RET(CmdDispatch);
    RET(CmdPipelineBarrier);
    RET(CmdCopyBuffer);
    return devData(dev)->gdpa(dev, p);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetInstanceProcAddr(VkInstance inst,
                                                            const char* p) {
    RET(GetInstanceProcAddr);
    RET(CreateInstance);
    RET(DestroyInstance);
    RET(CreateDevice);
    RET(GetDeviceProcAddr);
    RET(DestroyDevice);
    RET(QueueSubmit);
    RET(QueueWaitIdle);
    RET(DeviceWaitIdle);
    RET(WaitForFences);
    RET(CmdDispatch);
    RET(CmdPipelineBarrier);
    RET(CmdCopyBuffer);
    if (inst == VK_NULL_HANDLE) return nullptr;
    return instData(inst)->gipa(inst, p);
}
#undef RET

}  // namespace

extern "C" {

GCP_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vkGetInstanceProcAddr(VkInstance inst, const char* p) {
    return GetInstanceProcAddr(inst, p);
}
GCP_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vkGetDeviceProcAddr(VkDevice dev, const char* p) {
    return GetDeviceProcAddr(dev, p);
}
GCP_EXPORT VKAPI_ATTR VkResult VKAPI_CALL
vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface* v) {
    if (!v || v->sType != LAYER_NEGOTIATE_INTERFACE_STRUCT)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (v->loaderLayerInterfaceVersion > 2) v->loaderLayerInterfaceVersion = 2;
    v->pfnGetInstanceProcAddr = GetInstanceProcAddr;
    v->pfnGetDeviceProcAddr = GetDeviceProcAddr;
    v->pfnGetPhysicalDeviceProcAddr = nullptr;
    return VK_SUCCESS;
}

}  // extern "C"
