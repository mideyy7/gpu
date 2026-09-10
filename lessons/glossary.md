# Glossary

Every term used in the lessons, in one place. Grouped, not alphabetical, so
related ideas sit together.

---

## The project

| Term | Meaning |
|---|---|
| **N-body simulation** | A simulation of N objects that all interact — here, N particles pulling on each other by gravity. |
| **Particle** | A point with position, velocity, and mass. 32 bytes: `px py pz mass` + `vx vy vz pad`. |
| **Force calculation** | For each particle, sum the gravitational acceleration from every other particle. O(N²). Done by `force_calc.comp`. |
| **Integration** | Advance each particle one time step using its acceleration. Done by `integrate.comp`. |
| **Semi-implicit (symplectic) Euler** | Integration where velocity is updated first, then position uses the *new* velocity. Conserves energy well. |
| **Softening** | Adding `softening²` to distance-squared so two close particles don't produce infinite force. |
| **Readback** | Copying results from GPU memory back to CPU-readable memory. |
| **Step** | One force + integrate + (optional) readback. |
| **Batch** | K steps recorded into one command buffer and issued with one submit (`runBatch(K)`, `--batch K`). |
| **Sweep** | Running the sim at many particle counts to study scaling (`--sweep`). |
| **Warm-up** | Discarded steps run before measurement so the first (slow, cold) submissions don't skew results. |

## Profiling

| Term | Meaning |
|---|---|
| **Profiler** | A tool that measures *where* time goes, not just how much. |
| **Stage** | One measurable segment of GPU work: force dispatch, barrier stall, integrate dispatch, barrier stall, readback copy. |
| **`StepTimings`** | The struct holding one step's 6 raw ticks + 5 derived stage durations + 3 CPU-side durations. |
| **Wall-clock time** | Real elapsed time (as a stopwatch would measure), vs. CPU or GPU busy time. |
| **Overhead** | Here: `frame_total − gpu_total` — wall time in a step *not* spent doing GPU work. |
| **p50 / p99 (percentile)** | p50 = median (half of samples are faster). p99 = only 1% of samples are slower — surfaces the tail. |
| **Reconciliation** | Checking the 5 stage means sum to the GPU-total mean ("100% accounted"). A self-consistency check on the timestamps. |
| **ns / µs / ms** | nanosecond (10⁻⁹ s) / microsecond (10⁻⁶ s) / millisecond (10⁻³ s). 1000 ns = 1 µs, 1000 µs = 1 ms. |

## GPU & general

| Term | Meaning |
|---|---|
| **GPU** | Graphics Processing Unit — thousands of small cores, great at doing the same operation over big arrays. |
| **Compute (shader)** | A general-purpose GPU program that reads/writes arrays, no graphics involved. |
| **Headless** | No window, no image on screen. This project is headless compute. |
| **Driver** | Vendor software that actually talks to the GPU hardware. |
| **SIMD / parallelism** | "Same instruction, multiple data" — the GPU runs one program over many data elements at once. |
| **O(N²)** | Work grows with the square of N. Double N → 4× the work. |

## Vulkan objects

| Term | Meaning |
|---|---|
| **Vulkan** | A low-level, explicit cross-vendor GPU API (C functions like `vkCreateBuffer`). |
| **Loader** | The `libvulkan` library between your app and the driver; routes calls and inserts layers. |
| **ICD** (Installable Client Driver) | A driver the loader can use. MoltenVK is an ICD. |
| **Layer** | An optional plugin in the call chain (validation layer; this project's `VK_LAYER_gcp_profiler`). |
| **Validation layer** | `VK_LAYER_KHRONOS_validation` — checks every call for correct API usage and reports violations (VUIDs). |
| **VUID** | "Valid Usage ID" — an identifier for a specific rule the validation layer checks. "0 VUID" = clean. |
| **MoltenVK** | An ICD that translates Vulkan → Apple Metal, so Vulkan runs on macOS. Adds overhead; timings "not native." |
| **`VkInstance`** | Your connection to the loader. One per program. |
| **`VkPhysicalDevice`** | A GPU Vulkan can see. You query its properties and pick one. |
| **Queue family** | A group of GPU queues with certain capabilities (graphics / compute / transfer). |
| **`VkQueue`** | Where you submit work. Obtained from a family of the logical device. |
| **`VkDevice`** (logical device) | Your usable connection to one physical device; first arg to almost every Vulkan call. |
| **`VkCommandPool`** | Allocator that command buffers come from; tied to one queue family. |
| **`VkCommandBuffer`** | A recorded list of GPU commands. Record it, submit it, (optionally reset and re-record). |
| **Primary command buffer** | A top-level command buffer you submit directly (vs. secondary, which is called from a primary). |
| **`VkBuffer`** | A description of a chunk of GPU-accessible memory (size + allowed uses). Needs memory bound to it. |
| **`VkDeviceMemory`** | An actual GPU memory allocation. Bound to a buffer with `vkBindBufferMemory`. |
| **`VkFence`** | GPU→CPU signal: "this submission finished." CPU waits on it with `vkWaitForFences`. |
| **`VkSemaphore`** | GPU queue→GPU queue synchronisation. Not used here (one queue). |
| **`VkQueryPool`** | An array of slots the GPU writes measurements into. Here: `VK_QUERY_TYPE_TIMESTAMP`. |
| **`VkShaderModule`** | Compiled SPIR-V wrapped in a handle. |
| **`VkPipelineLayout`** | Declares a pipeline's interface: which descriptor set layouts + push-constant range. |
| **`VkPipeline`** (compute) | A fully baked GPU program ready to run — shader + layout. |
| **`VkDescriptorSetLayout`** | The *shape* of a descriptor set (e.g. "2 storage buffers for the compute stage"). |
| **`VkDescriptorPool`** | Allocator that descriptor sets come from. |
| **`VkDescriptorSet`** | An instance of a set layout with actual buffers plugged into its slots. |
| **`VkDebugUtilsMessengerEXT`** | The object that routes validation-layer messages to your callback. |

## Vulkan concepts

| Term | Meaning |
|---|---|
| **Handle** | An opaque identifier (usually a 64-bit value or pointer) Vulkan gives you for an object; you never look inside. |
| **`VK_NULL_HANDLE`** | Vulkan's null value for a handle. |
| **`sType`** | The first field of every Vulkan create-info struct; a constant identifying the struct type (for `pNext` extension chains). |
| **`pNext`** | A pointer chain letting structs carry extra extension data. Not used in this project (but every `sType` still set). |
| **Extension** | Optional Vulkan functionality you opt into by name (e.g. `VK_EXT_debug_utils`, `VK_KHR_portability_subset`). |
| **`vkGetInstanceProcAddr` / `vkGetDeviceProcAddr`** | Fetch a function pointer by name at runtime (needed for extension functions and for layers). |
| **Dispatch** | `vkCmdDispatch(gx,gy,gz)` — run the bound compute shader `gx*gy*gz` workgroups. |
| **Workgroup** | A group of shader invocations that run together; size declared in the shader (`local_size_x = 64`). |
| **Invocation** | One "thread" of a compute shader — handles one data element (here, one particle). |
| **`gl_GlobalInvocationID`** | Built-in shader input; `.x` is this invocation's global index (0 … total−1). |
| **Descriptor** | The binding between a shader's resource slot (`binding = 0`) and an actual buffer. |
| **Push constant** | A tiny block of data (here 16 bytes) shoved straight into the command buffer, faster than a descriptor for small data. |
| **`DEVICE_LOCAL`** | Fast memory on the GPU; shaders should use it; often not CPU-mappable. |
| **`HOST_VISIBLE`** | Memory the CPU can `vkMapMemory` and read/write directly. |
| **`HOST_COHERENT`** | A HOST_VISIBLE type where CPU and GPU automatically see each other's writes (no manual flush). |
| **Staging buffer** | A temporary HOST_VISIBLE buffer used to copy data into DEVICE_LOCAL memory. |
| **Mapping** | `vkMapMemory` — get a plain `void*` into HOST_VISIBLE memory. "Persistent mapping" = map once, keep the pointer. |
| **`timestampPeriod`** | Nanoseconds per GPU timestamp tick. Multiply tick differences by this. |
| **`timestampValidBits`** | Per queue family; how many bits of the timestamp counter are valid. 0 = timestamps unsupported. |

## Synchronisation

| Term | Meaning |
|---|---|
| **Hazard** | A situation where out-of-order/parallel execution would corrupt data. |
| **RAW** (read-after-write) | B reads what A wrote; B must wait for A. (integrate reads accel that force wrote.) |
| **WAR** (write-after-read) | B overwrites what A still needs to read. |
| **WAW** (write-after-write) | Two writes whose order matters. |
| **Pipeline barrier** | `vkCmdPipelineBarrier` — orders GPU work and makes writes visible across a hazard. |
| **Pipeline stage** | *When* in the GPU pipeline: `TOP_OF_PIPE`, `COMPUTE_SHADER`, `TRANSFER`, `BOTTOM_OF_PIPE`. |
| **Access mask** | *What memory operation*: `SHADER_READ`, `SHADER_WRITE`, `TRANSFER_READ`, `TRANSFER_WRITE`. |
| **Execution dependency** | The "wait for stage X before stage Y" part of a barrier. |
| **Memory dependency** | The "and make these writes visible to those reads" part of a barrier. |
| **`VkMemoryBarrier`** | A global memory barrier (covers all memory). Used here. |
| **`VkBufferMemoryBarrier`** | A memory barrier scoped to one buffer range. Tighter; not used here. |
| **Barrier stall** | GPU time spent waiting at a barrier — measured as `ticks[2] − ticks[1]` etc. |

## SPIR-V / GLSL

| Term | Meaning |
|---|---|
| **GLSL** | The C-like language the `.comp` shader files are written in. |
| **SPIR-V** | The portable binary format all Vulkan drivers accept. `.spv` files. |
| **`glslc`** | The compiler (from the Vulkan SDK) that turns GLSL into SPIR-V. Run by CMake at build time. |
| **`std430`** | A memory layout rule for storage buffers; determines byte offsets of struct fields. |
| **`vec3` / `vec4`** | GLSL 3- / 4-component float vectors. |
| **Swizzle** | `v.xyz`, `v.w` — accessing a subset of a vector's components. |
| **`inversesqrt(x)`** | GLSL built-in for 1/√x, one fast instruction. |

## C++

| Term | Meaning |
|---|---|
| **Header (`.h`) / source (`.cpp`)** | Declarations vs. definitions. `#include` pastes a header in. |
| **Translation unit** | One `.cpp` plus everything it includes; compiled independently. |
| **`#pragma once`** | Include the header at most once per translation unit. |
| **Linker** | Joins compiled object files (and libraries) into the final binary. |
| **`struct` / `class`** | Same thing; `struct` members public by default, `class` private by default. |
| **`public:` / `private:`** | Who may access a member. |
| **Pointer (`T*`)** | Holds the address of a value. `&x` = address of, `*p` = value at, `nullptr` = points at nothing. |
| **Reference (`T&`)** | An alias for an existing object; can't be null or repointed. `const T&` = read-only, no copy. |
| **`->` vs `.`** | `p->m` for a pointer (= `(*p).m`), `o.m` for an object. |
| **`std::vector<T>`** | A growable array. `.size()`, `.data()`, `.push_back()`, `.back()`. |
| **`std::array<T,N>`** | A fixed-size array; size known at compile time. |
| **`std::string`** | An owning text object; `+` concatenates; `.c_str()` → `const char*`. |
| **`auto`** | "Infer the type from the initialiser." |
| **Lambda** | An inline anonymous function; `[&](args){...}`. `[&]` = capture surrounding variables by reference. |
| **RAII** | "Resource Acquisition Is Initialization" — a resource is released by an object's destructor. `std::vector`, `std::lock_guard`. |
| **`const`** | "Does not change." After a method's `()`, "does not modify the object." |
| **`constexpr`** | "Known at compile time." |
| **`static_cast<T>(x)`** | Ordinary type conversion. |
| **`reinterpret_cast<T>(x)`** | "Treat these bits as type T." For pointers; dangerous; used for function-pointer and raw-memory casts. |
| **Exception** | `throw` jumps to the nearest matching `catch`. `std::runtime_error` carries a message via `.what()`. |
| **`std::chrono`** | Time library. `steady_clock`, `time_point`, `duration<double, std::nano>`, `.count()`. |
| **`std::atomic<T>`** | An integer safe to increment from multiple threads. `.fetch_add(1, memory_order_relaxed)`. |
| **`std::mutex` / `std::lock_guard`** | A lock, and an RAII helper that holds it for a scope. |
| **`std::mt19937`** | A seeded pseudo-random engine. Same seed → same sequence. |
| **`printf` format specs** | `%d` int, `%u` unsigned, `%f` float, `%s` string, `%zu` size_t, `%.3e` scientific, `%-5u` left-justified width 5, `%%` literal `%`. |
| **`argc` / `argv`** | `main`'s arguments: count, and array of C strings (`argv[0]` = program name). |
| **`EXIT_SUCCESS` / `EXIT_FAILURE`** | Standard process return codes: 0 and non-zero. |
| **Anonymous namespace** | `namespace { ... }` — makes its contents private to that `.cpp` file. |
| **PFN_vk...** | Vulkan's typedefs for "pointer to a function with this Vulkan signature." |
| **`VKAPI_ATTR` / `VKAPI_CALL`** | Macros that fix the calling convention of Vulkan callbacks and layer functions. |

## Build

| Term | Meaning |
|---|---|
| **CMake** | A build-system generator. `cmake -S . -B build` configures; `cmake --build build` builds. |
| **Target** | A thing CMake builds: an executable (`profiler`) or a library (`VkLayer_gcp_profiler`). |
| **`find_package` / `find_program`** | Locate a library / executable on the system. |
| **`add_custom_command`** | Teach the build a rule ("this input → run this command → that output"). Used for `glslc`. |
| **`target_compile_definitions`** | Pass a `#define` into the compile (`SHADER_DIR="..."`). |
| **Shared library** | `.dylib` / `.so` / `.dll` — loaded at runtime. The layer is one. |
| **Generator expression** (`$<...>`) | A CMake value known only at build time (e.g. `$<TARGET_FILE:...>`). |
| **`configure_file`** | Copy a template, substituting `@VAR@` placeholders. |
| **Layer manifest** | The JSON file the Vulkan loader reads to discover a layer (name + `library_path`). |
| **`VK_ADD_LAYER_PATH`** | Env var: *add* a directory to the loader's layer search path (keeps defaults). |
| **`VK_INSTANCE_LAYERS`** | Env var: force-enable named layers. |
| **`DYLD_LIBRARY_PATH`** | macOS env var: extra directories to search for shared libraries at runtime. |
