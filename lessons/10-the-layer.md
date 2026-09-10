# Part 10 — `layer/gcp_layer.cpp`: a reusable Vulkan layer

Files: `layer/gcp_layer.cpp` (308 lines), `layer/VkLayer_gcp_profiler.json.in`
(the manifest template).

This is a **completely separate build product** from `profiler`. It links no
project code and no `libvulkan` — only Vulkan's headers. It's a shared
library (`.dylib`) the Vulkan loader loads at runtime.

**What it does:** for *any* Vulkan application, count `vkQueueSubmit` calls
and the compute-relevant recorded commands (`vkCmdDispatch`,
`vkCmdPipelineBarrier`, `vkCmdCopyBuffer`), time the CPU inside
`vkQueueSubmit` and inside the fence/idle waits, and print a summary when
the app destroys its device. It's the "app-agnostic half" of the built-in
instrumentation — it can't know a dispatch is "the force pass," so it
reports counts and CPU timing, not the per-named-stage GPU breakdown.

---

## 1. What a Vulkan layer *is*

Recall the call chain from Part 02:

```
app  ─►  loader  ─►  [layer A]  ─►  [layer B]  ─►  driver
```

A layer is a shared library that inserts itself into that chain. When the
app calls `vkQueueSubmit`, the call goes: loader → **your layer's**
`QueueSubmit` → the next thing down the chain (another layer, or the
driver). Your layer can do work before and after calling down.

For the loader to use a layer it needs:

1. **A manifest** — a JSON file with the layer's name, its `library_path`,
   and the API version. The loader scans known directories for these.
2. **Three exported entry points** in the library:
   - `vkGetInstanceProcAddr` — "given a function name, give me your version
     of it (or pass through)."
   - `vkGetDeviceProcAddr` — same, for device-level functions.
   - `vkNegotiateLoaderLayerInterfaceVersion` — "let's agree on the
     loader↔layer protocol version."

Everything else in the file is machinery to make those three work.

### How it's activated

`scripts/run_layer.sh`:

```bash
export VK_ADD_LAYER_PATH="$BUILD"                    # where to find the manifest
export VK_INSTANCE_LAYERS="VK_LAYER_gcp_profiler"    # turn it on
```

- `VK_ADD_LAYER_PATH` is **additive** — it adds `build/` to the loader's
  search path *without* removing the default paths, so the validation layer
  is still found. (`VK_LAYER_PATH`, which *replaces* the path, was the cause
  of an early "validation layer not found" bug.)
- `VK_INSTANCE_LAYERS` names layers to force-enable even though the app
  didn't ask for them.

---

## 2. The two problems every layer must solve

### Problem A: "which function do I call down to?"

Your layer's `QueueSubmit` must eventually call the *real* `vkQueueSubmit`
(the next link in the chain). It can't call the global `vkQueueSubmit`
symbol — that would loop straight back to the loader and into your layer
again forever.

**Solution: dispatch tables.** During `CreateInstance` / `CreateDevice`, the
loader hands your layer a `pfnNextGetInstanceProcAddr` /
`pfnNextGetDeviceProcAddr`. You call *those* with each function name you
care about, and store the results — the genuine "next in chain" pointers —
in a struct. `gcp_layer.cpp` has two such structs:

```cpp
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
    // ... plus the counters and accumulators ...
};
```

`PFN_vkQueueSubmit` is a typedef Vulkan provides for "pointer to a function
with `vkQueueSubmit`'s signature."

### Problem B: "which instance/device am I in?"

`vkCmdDispatch` gets a `VkCommandBuffer`, not a `DeviceData*`. How does your
intercepted `CmdDispatch` find the counters for the right device?

**Solution: the dispatch-key trick.** Every *dispatchable* Vulkan handle
(`VkInstance`, `VkDevice`, `VkQueue`, `VkCommandBuffer`) is really a pointer
whose **first word** is a pointer the loader uses for dispatch. All handles
belonging to the same device share the same first word. So:

```cpp
void* dispatchKey(void* h) { return *reinterpret_cast<void**>(h); }
```

"Interpret the handle as a pointer-to-pointer, dereference once, get the
dispatch key." Then two hash maps keyed by that:

```cpp
std::unordered_map<void*, InstanceData*> g_inst;
std::unordered_map<void*, DeviceData*>   g_dev;

DeviceData* devData(void* h) {
    std::lock_guard<std::mutex> lk(g_mapMtx);
    return g_dev.at(dispatchKey(h));
}
```

So `devData(commandBuffer)` and `devData(queue)` and `devData(device)` all
return the same `DeviceData*`. The maps are guarded by a mutex because
`CreateDevice` (writing) and `CmdDispatch` (reading) can race.

(This is the same technique the official Vulkan sample layers use. In an
interview it's fine to say "I keyed per-device state off the handle's
dispatch pointer, the way the LunarG sample layers do.")

---

## 3. `CreateInstance` — hooking into the chain

```cpp
VKAPI_ATTR VkResult VKAPI_CALL CreateInstance(const VkInstanceCreateInfo* ci,
                                             const VkAllocationCallbacks* alloc,
                                             VkInstance* out) {
    auto* link = findInstLink(ci);
    auto gipa = link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    const_cast<VkLayerInstanceCreateInfo*>(link)->u.pLayerInfo = link->u.pLayerInfo->pNext;

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
```

Step by step:

- `findInstLink(ci)` walks the `pNext` chain of the create-info looking for
  the `VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO` link with
  `function == VK_LAYER_LINK_INFO`. That link carries
  `pfnNextGetInstanceProcAddr` — the "get proc addr" for the next thing in
  the chain.
- `link->...->pfnNextGetInstanceProcAddr` — grab it (`gipa`).
- `link->u.pLayerInfo = link->u.pLayerInfo->pNext` — **advance the chain**
  so the *next* layer down sees *its* link, not ours. (The `const_cast`
  strips `const` because we must edit this structure — the loader expects
  layers to do this.)
- `gipa(VK_NULL_HANDLE, "vkCreateInstance")` — ask the next layer for its
  `vkCreateInstance`, and call it. Now the real instance exists in `*out`.
- Create an `InstanceData`, store `gipa` and the next `DestroyInstance`, and
  register it in `g_inst` under the new instance's dispatch key.

`findInstLink`:

```cpp
const VkLayerInstanceCreateInfo* findInstLink(const VkInstanceCreateInfo* ci) {
    auto* p = static_cast<const VkLayerInstanceCreateInfo*>(ci->pNext);
    while (p && !(p->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO &&
                  p->function == VK_LAYER_LINK_INFO))
        p = static_cast<const VkLayerInstanceCreateInfo*>(p->pNext);
    return p;
}
```

A linked-list walk: follow `pNext` until you find the link with the right
`sType` and `function`.

`findDevLink` is the same for devices.

---

## 4. `CreateDevice` — building the device dispatch table

```cpp
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
```

- Same chain dance, but device create-info has **both** a next-`gipa` and a
  next-`gdpa` (device-level proc addr).
- Call the next layer's `vkCreateDevice`.
- Build the dispatch table: the `H(name)` macro expands to
  `dd->name = (PFN_vkname)gdpa(*out, "vkname")` — `#name` stringifies,
  `##` token-pastes. So `H(QueueSubmit)` becomes
  `dd->QueueSubmit = (PFN_vkQueueSubmit)gdpa(*out, "vkQueueSubmit");` —
  the genuine next-in-chain `vkQueueSubmit`.
- Register in `g_dev`, print a banner.

`#define`/`#undef` around a repetitive block is a common trick to avoid
typing the same pattern eight times.

---

## 5. The intercepted functions

### Counting commands

```cpp
VKAPI_ATTR void VKAPI_CALL CmdDispatch(VkCommandBuffer cb, uint32_t x, uint32_t y, uint32_t z) {
    DeviceData* d = devData(cb);
    d->dispatches.fetch_add(1, std::memory_order_relaxed);
    d->CmdDispatch(cb, x, y, z);            // call down to the real one
}
```

Look up the `DeviceData` from the command buffer's dispatch key, bump an
atomic counter, then call the real function. `CmdPipelineBarrier` and
`CmdCopyBuffer` are identical in shape.

### Timing the submit

```cpp
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
```

Bracket the real call with `Clock::now()`, add the elapsed nanoseconds to
`cpuSubmitNs` (under `d->mtx` because a plain `double +=` isn't atomic).
`fetch_add(c, ...)` adds `c` because one `vkQueueSubmit` can carry multiple
`VkSubmitInfo`.

### Timing the waits (a template)

```cpp
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
                                            const VkFence* fences, VkBool32 all, uint64_t to) {
    DeviceData* d = devData(dev);
    return timedWait(d, [&] { return d->WaitForFences(dev, c, fences, all, to); });
}
```

`timedWait` is a **function template**: `template <class Fn>` means "works
with any callable type `Fn`." `Fn&& fn` is a "forwarding reference" —
whatever lambda you pass, it's captured efficiently. The three wait-style
functions (`WaitForFences`, `QueueWaitIdle`, `DeviceWaitIdle`) each pass a
tiny lambda that calls their real counterpart, so the count+time logic isn't
copy-pasted three times.

---

## 6. `GetInstanceProcAddr` / `GetDeviceProcAddr` — the dispatch

```cpp
#define RET(name) if (!std::strcmp(p, "vk" #name)) return reinterpret_cast<PFN_vkVoidFunction>(name)

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
    return devData(dev)->gdpa(dev, p);        // everything else: pass through
}
```

When the loader (on the app's behalf) asks "layer, what's your
`vkQueueSubmit`?", this returns a pointer to the layer's `QueueSubmit`. For
any function the layer *doesn't* intercept, it returns the next layer's
version (`gdpa(dev, p)`), so those calls skip the layer entirely — no
overhead.

`RET(name)` expands to "if the requested name is `vkname`, return our
`name`." `GetInstanceProcAddr` is the same idea plus the instance-level and
create functions.

---

## 7. `printReport` and `DestroyDevice`

```cpp
VKAPI_ATTR void VKAPI_CALL DestroyDevice(VkDevice dev, const VkAllocationCallbacks* alloc) {
    DeviceData* dd = devData(dev);
    printReport(dd);                          // <-- the output happens here
    auto fn = dd->DestroyDevice;
    { std::lock_guard<std::mutex> lk(g_mapMtx); g_dev.erase(dispatchKey(dev)); }
    delete dd;
    fn(dev, alloc);
}
```

When the app tears down its device, the layer prints its accumulated
counters to `stderr`, unregisters and frees its `DeviceData`, then calls the
real `vkDestroyDevice`. `printReport` just `fprintf`s the numbers:

```
=== VK_LAYER_gcp_profiler report ===
  submits .............. 73
  vkCmdDispatch ....... 144   (1.97 / submit)
  vkCmdPipelineBarrier  144   (1.97 / submit)
  vkCmdCopyBuffer ..... 73
  fence / idle waits .. 74
  CPU in vkQueueSubmit  7062.0 us total   96.7 us / submit
  CPU in waits ........ 115953.4 us total   1566.9 us / wait
```

---

## 8. The exported entry points (bottom of file)

```cpp
extern "C" {

GCP_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vkGetInstanceProcAddr(VkInstance inst, const char* p) { return GetInstanceProcAddr(inst, p); }

GCP_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vkGetDeviceProcAddr(VkDevice dev, const char* p) { return GetDeviceProcAddr(dev, p); }

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
```

- `extern "C"` — disable C++ name mangling so the loader can find these
  symbols by their plain names (`vkGetInstanceProcAddr`, not
  `_Z21vkGetInstanceProcAddr...`).
- `GCP_EXPORT` — the "visible in the shared library" attribute (the rest of
  the file is compiled hidden, per `CXX_VISIBILITY_PRESET hidden` in
  CMake).
- `vkNegotiateLoaderLayerInterfaceVersion` — the handshake. The loader
  passes a struct; the layer caps the version at 2 (what it implements) and
  fills in its two proc-addr functions.

---

## 9. What the layer proves (and doesn't)

**Proves:** the instrumentation approach generalises — you can measure
submit/command structure and CPU costs of an app you didn't compile.
Running the *unmodified* `profiler` binary under the layer, the layer's
command counts exactly match the recorded structure (2 dispatch + 2 barrier
+ 1 copy per step), and its "CPU in waits / wait" (~1567 µs) matches the
in-app `fence wait (CPU)` mean (~1542 µs) within a few percent. Run with
`--batch 32` and the layer sees submits and copies drop 73 → 11 —
independent corroboration of the bottleneck fix.

**Doesn't:** give GPU per-stage timing. To bracket GPU time the layer would
have to inject its own timestamp command buffers into someone else's
submissions and harvest them later — doable but fiddly and a rich source of
validation errors, so it was deliberately left out. The per-named-stage GPU
breakdown stays with the in-app path, which *knows* which dispatch is which.

---

## Explain it in one sentence

> The layer is a shared library that inserts itself into the Vulkan call
> chain: on `CreateInstance`/`CreateDevice` it grabs the "next in chain"
> function pointers into a per-device dispatch table keyed off the handle's
> dispatch pointer, then its intercepted `vkQueueSubmit` / fence-waits /
> `vkCmd*` functions bump atomic counters and `std::chrono`-time the real
> call before passing it down, and on `vkDestroyDevice` it prints the
> totals — giving an app-agnostic view of submit and command structure and
> CPU cost for any Vulkan program.

## Questions you should be able to answer

1. What three functions must a layer export, and what is the JSON manifest
   for?
2. Why can't the layer's `QueueSubmit` just call the global
   `vkQueueSubmit`? What does it call instead, and where did that pointer
   come from?
3. What is the "dispatch key" trick and what problem does it solve?
4. Why `VK_ADD_LAYER_PATH` instead of `VK_LAYER_PATH` in the run script?
5. What does the chain-advance line
   (`link->u.pLayerInfo = link->u.pLayerInfo->pNext`) do, and why is a
   `const_cast` needed?
6. Why are the counters `std::atomic` but the `double` accumulators guarded
   by a mutex?
7. What does running `profiler` under the layer independently confirm about
   the Phase 8 fix?
8. Why doesn't the layer report per-stage GPU time?
