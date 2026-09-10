# Part 01 — C++ for this codebase

This is not a full C++ course. It's *exactly* the subset used in this
project, each feature pinned to a real line so you can see it in context.
Read with the files open.

C++ is a language where **you manage everything explicitly**: memory,
resource lifetimes, when things are copied. That verbosity is why the code
looks long, but almost every line is doing one small, nameable thing.

---

## 1. Files: headers, source files, and `#include`

C++ splits code into two kinds of file:

- **Header** (`.h`) — *declarations*: "here is a thing called `Simulation`,
  it has these members and these functions." No actual logic (mostly).
- **Source** (`.cpp`) — *definitions*: the actual function bodies.

Other files that want to use `Simulation` do `#include "simulation.h"`.
`#include` literally means "paste the contents of that file here." So a
`.cpp` file plus everything it includes forms one **translation unit**, and
each translation unit is compiled independently into an object file, then
the **linker** stitches the object files together into the final program.

`CMakeLists.txt` lists the four `.cpp` files that make the `profiler` binary
(`main.cpp`, `vk_context.cpp`, `simulation.cpp`, `report.cpp`). Each is
compiled separately; the linker joins them.

### `#pragma once`

`vk_context.h:1`, `simulation.h:1`, `report.h:1` all start with:

```cpp
#pragma once
```

If two files both `#include "simulation.h"`, without protection the compiler
would see the `Simulation` class defined twice and error. `#pragma once`
means "if you've already included me in this translation unit, skip me." It
is a header's way of saying "include me as many times as you like, I'll only
take effect once."

### `#include <...>` vs `#include "..."`

- `#include <vulkan/vulkan.h>` — angle brackets: a **system / library**
  header, found on the compiler's search path.
- `#include "simulation.h"` — quotes: a header **in this project**, found
  next to the current file.

### The standard-library headers used here

| Header | Gives you | Seen in |
|---|---|---|
| `<cstdio>` | `printf`, `fprintf`, `snprintf`, `FILE`, `fopen` | everywhere |
| `<cstdlib>` | `EXIT_SUCCESS`, `std::exit`, `std::strtoul` | `main.cpp` |
| `<cstring>` | `std::strcmp`, `std::strstr`, `std::memcpy` | `vk_context.cpp`, `simulation.cpp` |
| `<cstdint>` | fixed-width integer types `uint32_t`, `uint64_t` | headers |
| `<string>` | `std::string` | most files |
| `<vector>` | `std::vector` (growable array) | most files |
| `<array>` | `std::array` (fixed-size array) | `simulation.cpp` |
| `<cmath>` | `std::sqrt`, `std::fabs`, `std::lround` | `main.cpp`, `report.cpp` |
| `<algorithm>` | `std::sort`, `std::min`, `std::max` | `report.cpp`, `main.cpp` |
| `<chrono>` | clocks and time durations | `simulation.cpp` |
| `<random>` | random number generators | `main.cpp` |
| `<fstream>` | reading files (`std::ifstream`) | `vk_context.cpp` |
| `<stdexcept>` | `std::runtime_error` (an exception type) | several |
| `<atomic>` | `std::atomic` (thread-safe counters) | `layer/gcp_layer.cpp` |
| `<mutex>` | `std::mutex`, `std::lock_guard` | `layer/gcp_layer.cpp` |
| `<unordered_map>` | hash map | `layer/gcp_layer.cpp` |

The `c`-prefixed ones (`<cstdio>` etc.) are the C standard library made
available in the `std::` namespace. `std::printf` and `printf` are the same
function.

---

## 2. Primitive types

| Type | Meaning | Example line |
|---|---|---|
| `bool` | `true` / `false` | `simulation.h:36` `bool gpuValid = false;` |
| `int` | signed integer, ≥16 bits (32 in practice) | `main.cpp:313` `for (int i = 1; ...)` |
| `float` | 32-bit floating point (~7 digits precision) | `simulation.h:11` `float x = 0` |
| `double` | 64-bit floating point (~15 digits) | `simulation.h:38` `double timestampPeriodNs` |
| `char` | one byte, often a character | `vk_context.cpp:43` `char buf[128];` |
| `void` | "nothing" — a function that returns nothing, or `void*` = "pointer to untyped memory" | `simulation.h:62` `void destroy();` |

### Fixed-width integers (`<cstdint>`)

`uint32_t` = unsigned 32-bit integer (0 to ~4.29 billion).
`uint64_t` = unsigned 64-bit.
`size_t`   = unsigned integer big enough to hold any array size (usually 64-bit).

Vulkan uses these everywhere because a GPU API cannot afford ambiguity about
how big a number is. `simulation.h:22` `uint32_t particleCount = 0;`

### Why `double` for timings but `float` for physics?

- Physics (`Particle`, the shaders): `float`. GPUs are much faster at
  `float`, and particle positions don't need 15 digits.
- Timing math (`StepTimings`, `Report`): `double`. When you sum hundreds of
  nanosecond values and compute means, the extra precision avoids drift.

---

## 3. `const` and `constexpr`

### `const`

"This does not change." Applied to a variable, parameter, or (on the right
of a member function) to `*this`.

- `main.cpp:472` `const float pix = particles[i].px;` — a local you promise
  not to reassign.
- `vk_context.h:28` `... findMemoryType(...) const;` — the `const` after the
  parameter list means "this method does not modify the object it's called
  on." You can call it on a `const VulkanContext`.
- `report.cpp:6` `void Report::addSample(const StepTimings& t)` — `t` is a
  reference we won't modify (see references below).

### `constexpr`

"Known at compile time." `simulation.h:30-31`:

```cpp
static constexpr uint32_t kTimestampsPerStep = 6;
static constexpr uint32_t kMaxBatch = 64;
```

These are true constants — the compiler substitutes `6` and `64` directly.
`static` here (at file scope in a header) means "each translation unit gets
its own copy, don't complain about multiple definitions."

`6` is the number of timestamps written per simulation step (part 07). `64`
is the largest batch of steps the code will record into one command buffer.
The query pool is sized `6 * 64 = 384` (`simulation.cpp:97`).

---

## 4. `struct` and `class`

A `struct` and a `class` in C++ are **the same thing** with one difference:
`struct` members are **public by default**, `class` members are **private by
default**.

### `struct` used as a plain data bundle

`simulation.h:10-12`:

```cpp
struct Vec3 {
    float x = 0, y = 0, z = 0;
};
```

A `Vec3` is just three floats named `x`, `y`, `z`. `= 0` is a **default
member initializer**: if you write `Vec3 v;` you get `{0, 0, 0}` without
having to say so. `Vec3 v{1, 2, 3};` sets them explicitly.

`Particle` (`simulation.h:16-19`) and `SimParams` (`21-26`) and
`StepTimings` (`35-50`) are the same idea — bundles of named fields with
sensible defaults.

### `class` used as an object with behaviour

`simulation.h:58`:

```cpp
class Simulation {
   public:
    void init(...);
    void destroy();
    void runBatch(uint32_t steps);
    // ...
   private:
    void createBuffers(...);
    // ... plus a lot of member variables (the trailing underscore names)
};
```

- `public:` — anything can call these. This is the class's interface.
- `private:` — only `Simulation`'s own methods can touch these. Callers
  can't reach in and grab `particleBuffer_`.
- The trailing underscore (`ctx_`, `params_`, `particleBuffer_`) is a
  **naming convention** for "this is a member variable," so inside a method
  you can tell `params_` (the member) from a local called `params`.

### Member functions defined inline

`simulation.h:72`:

```cpp
void stepOnce() { runBatch(1); }
```

Short methods can have their body right in the header. `stepOnce()` is
literally just "call `runBatch` with 1." `lastTimings()` (`75`) and
`timestampsSupported()` (`77`) are the same — one-line accessors.

### `.` vs `->`

- `object.member` when you have the object directly: `params_.particleCount`.
- `pointer->member` when you have a pointer to it: `ctx_->device`
  (`ctx_` is a `VulkanContext*`). `ptr->x` is shorthand for `(*ptr).x`.

---

## 5. Zero-initialisation: `VkFoo x{};`

You'll see this constantly, e.g. `vk_context.cpp:52`:

```cpp
VkApplicationInfo appInfo{};
```

`VkApplicationInfo` is a plain C struct with ~5 fields. The `{}` means
"**value-initialise**": set every field to zero / null. Without it, the
fields would hold garbage (whatever was in that stack memory before), and
Vulkan would read the garbage and misbehave.

The pattern is always: zero the struct, then set the fields you care about:

```cpp
VkApplicationInfo appInfo{};
appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;   // required tag (see part 02)
appInfo.pApplicationName = "gpu-compute-profiler";
appInfo.apiVersion = VK_API_VERSION_1_2;
```

---

## 6. Pointers and references

### Pointer `T*`

A pointer is a variable holding **the address of** another value.

- `&x` — "address of `x`".
- `*p` — "the value `p` points at" (dereference).
- `nullptr` — "points at nothing." Always initialise pointers to this if you
  don't have a real target yet: `simulation.h:90` `VulkanContext* ctx_ = nullptr;`

Vulkan's C API is full of pointers because C has no other way to pass "give
me back a value" or "here is an array":

```cpp
// vk_context.cpp:83 -- &instance means "here's where to write the result"
check(vkCreateInstance(&instInfo, nullptr, &instance), "vkCreateInstance");
```

`vkCreateInstance` needs (a) a pointer to the settings struct, (b) an
allocator we don't use (`nullptr`), (c) `&instance` = the address of our
`VkInstance` variable, so the function can write the created handle there.

### The "count then data" pattern

`vk_context.cpp:101-105`:

```cpp
uint32_t deviceCount = 0;
vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);   // 1st call: how many?
std::vector<VkPhysicalDevice> devices(deviceCount);            // allocate that many
vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data()); // 2nd call: fill it
```

Call once with `nullptr` for the data pointer → Vulkan writes the *count*.
Allocate. Call again with a real pointer → Vulkan writes the *data*. This
pattern appears ~8 times in `vk_context.cpp` and `simulation.cpp`.
`.data()` returns a pointer to a `std::vector`'s underlying array.

### Reference `T&`

A reference is an **alias** — another name for an existing object. Unlike a
pointer, it can't be null and can't be "repointed."

- `simulation.cpp:61` `auto& ctx = *ctx_;` — `ctx` is now a short alias for
  `*ctx_` (the `VulkanContext` the pointer points at), so the rest of the
  function can write `ctx.device` instead of `ctx_->device`.
- `report.cpp:6` `void Report::addSample(const StepTimings& t)` —
  `const StepTimings&` means "pass `t` by reference (no copy) and promise
  not to modify it." Passing a big struct by value would copy all its
  fields; by `const&` is free.
- `simulation.cpp:106` `... VkBuffer& buf, VkDeviceMemory& mem)` — non-const
  references used as **output parameters**: the lambda writes the created
  buffer/memory back into the caller's variables.

Rule of thumb in this codebase:

- Small value you won't change → pass by value (`uint32_t n`).
- Big struct you won't change → `const T&`.
- Something the function must fill in → `T&` or `T*`.

---

## 7. `std::vector<T>` — the growable array

`std::vector<int>` is "a dynamically-sized array of `int`." `<T>` is a
**template parameter** — `vector` is a recipe, `vector<int>` is the version
for `int`.

Operations used here:

| Call | Meaning | Example |
|---|---|---|
| `std::vector<Particle> p(n)` | make one with `n` default elements | `main.cpp:21` |
| `.size()` | how many elements | `report.cpp:25` |
| `.data()` | pointer to the first element | `vk_context.cpp:105` |
| `.push_back(x)` | append `x`, growing if needed | `report.cpp:8` |
| `.back()` | last element (reference) | `simulation.h:75` |
| `.front()` | first element | `report.cpp:26` |
| `.empty()` | is size 0? | `report.cpp:21` |
| `.assign(n, val)` | resize to `n` copies of `val` | `simulation.cpp:402` |
| `v[i]` | element `i` (no bounds check) | everywhere |

### Range-based `for`

`main.cpp:22`:

```cpp
for (auto& q : p) {          // for each element q of vector p, by reference
    q.px = pos(rng);         // modifying q modifies the element in p
}
```

`report.cpp:24` `for (double x : v) sum += x;` — here `x` is a *copy* (no
`&`), fine for a cheap `double`.

`main.cpp:384` `for (const auto& c : fcases)` — `const auto&` = "each
element, by reference, read-only."

---

## 8. `std::string` and C strings

Two kinds of text in C++:

- **C string**: `const char*` — a pointer to a zero-terminated array of
  `char`. String literals like `"hello"` are C strings.
- **`std::string`**: a proper object that owns its characters, can grow,
  supports `+`.

`vk_context.cpp:190` `throw std::runtime_error("cannot open shader: " + path);`
— `"..." + path` works because `path` is a `std::string` and `+` is
concatenation.

`.c_str()` converts a `std::string` back to a `const char*` for functions
(like `printf`) that need one: `main.cpp:236`
`... ctx.deviceSummary().c_str()`.

`std::strcmp(a, b) == 0` (`vk_context.cpp:27`) compares two C strings; it
returns 0 when they're equal (not `1`/`true` — it returns the *difference*,
so `== 0` means "same").

---

## 9. `auto`

`auto` tells the compiler "figure out the type from the right-hand side."

- `simulation.cpp:61` `auto& ctx = *ctx_;` → `ctx` is `VulkanContext&`.
- `main.cpp:48` `auto particles = makeParticles(n, seed);` → `particles` is
  `std::vector<Particle>` (that's what `makeParticles` returns).
- `report.cpp:28` `auto pct = [&](double p) { ... };` → `pct` is "some
  lambda type" (you can't spell it, so `auto` is required).

Use it when the type is obvious or unspellable; write the type explicitly
when it aids clarity.

---

## 10. Lambdas (anonymous functions)

A **lambda** is a function you define inline, often to pass to another
function or to avoid repeating a few lines.

`simulation.cpp:106-123` — `makeBuffer` is a lambda that creates a Vulkan
buffer + its memory. It's used three times right below, so factoring it out
avoids 3× the boilerplate:

```cpp
auto makeBuffer = [&](VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags memFlags, VkBuffer& buf,
                      VkDeviceMemory& mem) {
    // ... 15 lines of Vulkan buffer creation ...
};

makeBuffer(particleBytes_, ...usage..., DEVICE_LOCAL, particleBuffer_, particleMemory_);
makeBuffer(accelBytes_,    ...usage..., DEVICE_LOCAL, accelBuffer_,    accelMemory_);
```

### The capture list `[&]`

The `[...]` at the start says what outside variables the lambda can see.

- `[&]` — "capture everything I use **by reference**." So inside
  `makeBuffer`, `ctx` refers to the same `ctx` as the enclosing function.
  This is the common choice here.
- `[]` — capture nothing (a pure function of its arguments).
- You can also list specific names, but this codebase just uses `[&]`.

### Lambda with an explicit return type

`main.cpp:315`:

```cpp
auto next = [&]() -> const char* {
    if (i + 1 >= argc) { ... std::exit(EXIT_FAILURE); }
    return argv[++i];
};
```

`-> const char*` after the `()` spells out the return type (a C string).
`next()` returns the next command-line argument and advances `i`.

---

## 11. `std::array<T, N>` — the fixed-size array

`simulation.cpp:161` `std::array<VkDescriptorSetLayoutBinding, 2> bindings{};`
— exactly 2 elements, size known at compile time, lives on the stack (no
heap allocation). Use it instead of `std::vector` when the count is fixed
and small. `{}` zero-initialises all elements.

---

## 12. Casts — converting between types

C++ has named casts so you can see *what kind* of conversion is happening.

| Cast | Use | Example |
|---|---|---|
| `static_cast<T>(x)` | ordinary, safe-ish conversions (number → number, up/down a class hierarchy) | `vk_context.cpp:78` `static_cast<uint32_t>(extensions.size())` |
| `reinterpret_cast<T>(x)` | "trust me, treat these bits as type T" — pointers only, dangerous | `vk_context.cpp:86` casting a function-pointer-of-unknown-type to `PFN_vkCreateDebugUtilsMessengerEXT` |
| C-style `(T)x` | old style, avoid in new code; a few appear in `printf` arg lists | `main.cpp:137` `(unsigned long long)t.rawTicks[i]` |

Why `static_cast<uint32_t>(v.size())`? `.size()` returns `size_t` (64-bit);
the Vulkan struct field wants `uint32_t` (32-bit). The cast says "yes, I
know I'm narrowing, it's fine, these counts are small."

`reinterpret_cast` example, `vk_context.cpp:86-87`:

```cpp
auto createFn = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
    vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
```

`vkGetInstanceProcAddr` returns a generic function pointer
(`PFN_vkVoidFunction`). We *know* it's really a
`vkCreateDebugUtilsMessengerEXT`, so we reinterpret the bits. (Details in
part 05.)

`simulation.cpp:427` `reinterpret_cast<const float*>(char_ptr + offset)` —
"the bytes at this address are actually an array of floats." Used to read
GPU results out of a raw memory mapping.

---

## 13. Resource management: create/destroy pairs and RAII

C++ has no garbage collector. Every resource you acquire, you must release.

### The manual pattern (used for Vulkan objects)

Vulkan objects come in `vkCreateX` / `vkDestroyX` pairs. `Simulation` holds
~15 of them as members, all initialised to `VK_NULL_HANDLE`
(`simulation.h:93-121`), created in `init()` and its helpers, and released
in `destroy()` (`simulation.cpp:441-464`):

```cpp
void Simulation::destroy() {
    if (!ctx_) return;                       // already destroyed / never inited
    VkDevice d = ctx_->device;
    vkDeviceWaitIdle(d);                     // wait for the GPU to finish first!
    if (readbackMapped_) { vkUnmapMemory(d, readbackMemory_); readbackMapped_ = nullptr; }
    if (loopFence_)  vkDestroyFence(d, loopFence_, nullptr);
    if (queryPool_)  vkDestroyQueryPool(d, queryPool_, nullptr);
    // ... every other object, each guarded by `if (handle)` ...
    ctx_ = nullptr;
}
```

The `if (handle)` guards make `destroy()` safe to call even if `init()`
failed halfway. `ctx_ = nullptr` at the end makes a second `destroy()` call
a no-op.

**Interview note:** a more idiomatic C++ approach wraps each handle in its
own small class whose *destructor* calls `vkDestroyX` (that's what "RAII" —
Resource Acquisition Is Initialization — means). This project uses one
explicit `destroy()` instead, for readability: all the teardown is in one
place you can read top to bottom. It's a deliberate, defensible trade-off,
and you should be ready to say so.

### RAII that *is* used

`std::vector`, `std::string`, `std::ifstream`, `std::lock_guard` all clean
themselves up when they go out of scope. E.g. `vk_context.cpp:189`
`std::ifstream file(...)` — the file is closed automatically when `file`'s
scope ends. `std::lock_guard<std::mutex> lk(m);` (`layer/gcp_layer.cpp`)
locks `m` and unlocks it automatically at the end of the block.

### `*this = VulkanContext{};`

`vk_context.cpp:164`. `VulkanContext{}` makes a fresh, all-default
`VulkanContext`; assigning it to `*this` resets every member to its default.
A tidy way to say "this object is now blank."

---

## 14. Exceptions: `throw` / `try` / `catch`

When something goes irrecoverably wrong deep in a call, instead of returning
an error code up through ten functions, you `throw`. Execution jumps to the
nearest matching `catch`.

`vk_context.cpp:41-47` — the `check` helper:

```cpp
void check(VkResult r, const char* what) {
    if (r != VK_SUCCESS) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "%s failed (VkResult %d)", what, (int)r);
        throw std::runtime_error(buf);
    }
}
```

Every Vulkan call is wrapped: `check(vkCreateInstance(...), "vkCreateInstance");`
If it fails, a `std::runtime_error` (a standard exception type that carries a
message) flies out.

`main.cpp:353-358` catches it:

```cpp
try {
    ctx.create(true);
} catch (const std::exception& e) {
    std::fprintf(stderr, "fatal: %s\n", e.what());
    return EXIT_FAILURE;
}
```

`std::exception` is the base type of all standard exceptions; catching it by
`const&` catches `runtime_error` and friends. `e.what()` returns the message
string. `EXIT_FAILURE` is a standard "the program failed" return code
(`EXIT_SUCCESS` is 0 = "fine").

---

## 15. `<chrono>` — measuring wall-clock time

`simulation.cpp:10-14`:

```cpp
using Clock = std::chrono::steady_clock;

static double nanosBetween(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::nano>(b - a).count();
}
```

- `using Clock = ...;` — a **type alias**. From now on `Clock` means
  `std::chrono::steady_clock`. (`using X = Y;` is the modern spelling of
  `typedef`.)
- `steady_clock` — a clock that only ever moves forward at a steady rate
  (unlike the wall clock, which can jump when the system time is adjusted).
  The right choice for measuring durations.
- `Clock::time_point` — "a moment in time" as reported by that clock.
- `Clock::now()` — the current moment.
- `b - a` — subtracting two time points gives a **duration**.
- `std::chrono::duration<double, std::nano>(b - a)` — re-express that
  duration as "a `double` count of nanoseconds."
- `.count()` — pull out the raw number.

Used in `runBatch` (`simulation.cpp:301, 379, 381, 384`) to time the CPU
side: how long `vkQueueSubmit` took, how long `vkWaitForFences` blocked.

---

## 16. `<random>` — deterministic random particles

`main.cpp:16-32`:

```cpp
std::vector<Particle> makeParticles(uint32_t n, uint32_t seed) {
    std::mt19937 rng(seed);                                   // the generator
    std::uniform_real_distribution<float> pos(-5.0f, 5.0f);   // shape the output
    std::uniform_real_distribution<float> mass(0.5f, 2.0f);
    std::uniform_real_distribution<float> vel(-1.0f, 1.0f);
    std::vector<Particle> p(n);
    for (auto& q : p) {
        q.px = pos(rng);  // pos(rng) draws one float in [-5, 5]
        // ...
    }
    return p;
}
```

- `std::mt19937` — the Mersenne Twister pseudo-random engine. `rng(seed)`
  seeds it. **Same seed → same sequence**, which is why the self-tests are
  reproducible.
- `std::uniform_real_distribution<float> pos(-5, 5)` — an object that maps
  the engine's raw bits into uniformly-distributed floats in `[-5, 5]`. You
  call it like a function: `pos(rng)`.

---

## 17. `<atomic>` and `<mutex>` — used only in the layer

The layer (`layer/gcp_layer.cpp`) can be called from multiple threads (the
app's threads), so its counters must be thread-safe.

- `std::atomic<uint64_t> submits{0};` — an integer that can be incremented
  from multiple threads without corruption.
- `d->submits.fetch_add(1, std::memory_order_relaxed);` — atomically add 1.
  `memory_order_relaxed` = "I only care that the count is right, not about
  ordering relative to other memory" (the fast, minimal guarantee).
- Reading it: `const uint64_t s = d->submits;` — an atomic converts to its
  value implicitly.
- `std::mutex mtx;` + `std::lock_guard<std::mutex> lk(d->mtx);` — for the
  `double` accumulators (which aren't atomic), take the lock for the
  duration of the block; it releases automatically (RAII again).

You can gloss over the memory-ordering nuance in an interview; the key
sentence is "the layer's counters are `std::atomic` because a Vulkan layer
can be entered from any of the application's threads."

---

## 18. `printf`-family formatting

`std::printf("format", args...)` writes to standard output; `std::fprintf(f,
...)` to a file or `stderr`; `std::snprintf(buf, size, ...)` into a string
buffer.

Format specifiers seen here:

| Spec | Means | Example |
|---|---|---|
| `%d` | signed int | `"(VkResult %d)"` |
| `%u` | unsigned int | `"N=%u"` |
| `%f` | float/double, fixed notation | `"%.1fus"` = one decimal place |
| `%.3e` | scientific notation, 3 decimals | `"rel-err=%.3e"` |
| `%s` | C string | `"%s\n", header.c_str()` |
| `%zu` | `size_t` | `"%zu frames"` |
| `%llu` | `unsigned long long` | `"%llu", (unsigned long long)x` |
| `%p` | pointer | (layer report) |
| `%-5u` | left-justified in a 5-wide field | table columns in `report.cpp` |
| `%10s` | right-justified in a 10-wide field | table headers |
| `%%` | a literal `%` sign | `"%7.1f%%"` |
| `\n` | newline | everywhere |

`report.cpp:69-70` builds the timing table purely with field-width specs so
the columns line up.

---

## 19. Preprocessor bits

`simulation.cpp:16-18`:

```cpp
#ifndef SHADER_DIR
#define SHADER_DIR "shaders"
#endif
```

`SHADER_DIR` is normally defined by the build system (`CMakeLists.txt` sets
it to the absolute path of the compiled-shader folder). This block says "if
nobody defined it, fall back to the literal `"shaders"`," so the file still
compiles outside CMake.

`layer/gcp_layer.cpp` has a similar guard defining `GCP_EXPORT` to a
compiler attribute that marks a function as visible in the shared library.

---

## 20. Small things you'll see and wonder about

- **Ternary `cond ? a : b`** — an expression that is `a` if `cond` is true,
  else `b`. `main.cpp:413` `failures == 0 ? "ALL PASS" : "FAILURES"`.
- **`sizeof(x)`** — the size of `x` in bytes, known at compile time.
  `simulation.cpp:103` `sizeof(Particle) * params_.particleCount` = total
  bytes for all particles.
- **`argc` / `argv`** — `main(int argc, char** argv)`. `argc` = argument
  count, `argv` = array of C strings. `argv[0]` is the program name,
  `argv[1]` onward are the flags you typed.
- **`VKAPI_ATTR` / `VKAPI_CALL`** — macros Vulkan puts on callback and layer
  functions to force a specific calling convention. Treat them as noise that
  must be present; they expand to nothing on most platforms.
- **`enum` flags and `|`** — Vulkan uses integers where each bit is a
  boolean option. `A_BIT | B_BIT` combines them; `flags & A_BIT` tests one;
  `flags |= A_BIT` adds one. E.g. `vk_context.cpp:113`
  `if (qProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT)` = "does this queue
  family have the compute bit set?"
- **Anonymous struct arrays** — `main.cpp:381-383`:
  ```cpp
  const struct { uint32_t n, seed; } fcases[] = {{4,1},{8,2},{16,3},{13,99}};
  ```
  "an array of little unnamed structs, each holding `n` and `seed`." A quick
  way to write a table of test cases.

---

## Explain it in one sentence

> The code uses a small, deliberate subset of C++: plain structs for data,
> one class with a manual `destroy()` for the GPU resources, `std::vector`
> for dynamic arrays, references and `const&` to avoid copies, lambdas to
> factor out repeated Vulkan boilerplate, exceptions for fatal errors, and
> `<chrono>` for CPU-side timing.

## Questions you should be able to answer

1. What's the difference between a `.h` and a `.cpp` file, and what does
   `#include` actually do?
2. Why does almost every Vulkan struct get created as `VkFoo x{};`?
3. What's the difference between a pointer and a reference? When does this
   code use each?
4. What does `[&]` mean at the start of a lambda?
5. Why is there a hand-written `destroy()` instead of destructors, and what
   would the "proper" C++ approach be?
6. What does `check(vkDoSomething(...), "vkDoSomething")` do on failure, and
   where is that caught?
7. Why `double` for timings but `float` for particle positions?
8. What is `std::mt19937` and why does seeding it matter for the tests?
