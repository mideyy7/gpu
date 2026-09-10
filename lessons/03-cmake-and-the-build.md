# Part 03 — CMake and the build

File under study: `CMakeLists.txt` (63 lines).

---

## 1. What problem CMake solves

You have `.cpp` files and `.comp` shader files. You need:

- each `.cpp` compiled with the right flags and include paths,
- the Vulkan library found and linked,
- each `.comp` compiled to `.spv` by `glslc`,
- the compiled shaders' location passed into the C++ as a `#define`,
- a second build product (the layer `.dylib`) with different settings,
- a generated JSON file next to it.

Doing that by hand with compiler commands is fragile and OS-specific.
**CMake** is a build-system generator: you describe the build in
`CMakeLists.txt`, then:

```
cmake -S . -B build      # "configure": read CMakeLists.txt, generate a real
                         #  build system (Makefiles) in the ./build folder
cmake --build build      # "build": run that build system
```

The `build/` folder is disposable — it's in `.gitignore`. Delete it and
re-run the two commands any time.

---

## 2. `CMakeLists.txt`, line by line

### Project setup

```cmake
cmake_minimum_required(VERSION 3.20)
project(gpu_compute_profiler CXX)
```

- `cmake_minimum_required` — refuse to run on an ancient CMake that lacks
  features used below.
- `project(name CXX)` — name the project; `CXX` = "this project is C++ only"
  (no C, no Fortran). Saves time by not probing for other compilers.

```cmake
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```

- Compile as **C++17**. (The code uses C++17 features like `std::lround` in
  `<cmath>`, structured-ish init, `if constexpr`-era library bits.)
- `REQUIRED ON` — error out rather than silently falling back to an older
  standard if the compiler doesn't support 17.

```cmake
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()
```

If the user didn't pick a build type, default to **Debug** (no
optimisation, assertions on, debug symbols). For a profiling project you'd
normally want `Release` for real numbers, but Debug is the safe default for
development and keeps the validation-layer workflow honest. You can override
with `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`.

### Finding tools and libraries

```cmake
find_package(Vulkan REQUIRED)
```

CMake ships a "Find module" for Vulkan. This locates the Vulkan headers and
the `libvulkan` library on the system and creates an **imported target**
called `Vulkan::Vulkan` that other targets can link against. `REQUIRED` =
fail configuration if Vulkan isn't found. It also sets
`Vulkan_INCLUDE_DIRS` (used later for the layer).

```cmake
find_program(GLSLC glslc REQUIRED)
```

Find the `glslc` executable (the GLSL→SPIR-V compiler from the Vulkan SDK)
and store its path in the variable `GLSLC`. `REQUIRED` = fail if missing.

### Shader compilation

```cmake
set(SHADER_SRC_DIR ${CMAKE_SOURCE_DIR}/shaders)
set(SHADER_OUT_DIR ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_OUT_DIR})
```

- `CMAKE_SOURCE_DIR` — the folder containing this `CMakeLists.txt`.
- `CMAKE_BINARY_DIR` — the `build/` folder.
- So source shaders live in `shaders/`, compiled `.spv` files will be
  written to `build/shaders/`. Make that output folder now.

```cmake
file(GLOB SHADER_SOURCES CONFIGURE_DEPENDS ${SHADER_SRC_DIR}/*.comp)
```

- `file(GLOB ...)` — find all files matching `shaders/*.comp` and put the
  list in `SHADER_SOURCES`. (Here: `force_calc.comp`, `integrate.comp`,
  `stub.comp`.)
- `CONFIGURE_DEPENDS` — normally CMake only globs at configure time, so
  adding a new shader file wouldn't be noticed until you re-ran `cmake -S .
  -B build`. This flag tells CMake to re-check the glob on every build. (It
  was added after a real "why isn't my new shader building" moment.)

```cmake
set(SPIRV_BINARIES "")
foreach(SHADER ${SHADER_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER} NAME)
    set(SPIRV_OUT ${SHADER_OUT_DIR}/${SHADER_NAME}.spv)
    add_custom_command(
        OUTPUT ${SPIRV_OUT}
        COMMAND ${GLSLC} ${SHADER} -o ${SPIRV_OUT}
        DEPENDS ${SHADER}
        COMMENT "Compiling shader ${SHADER_NAME} -> SPIR-V"
        VERBATIM
    )
    list(APPEND SPIRV_BINARIES ${SPIRV_OUT})
endforeach()
```

For each shader source:

- `get_filename_component(... NAME)` — strip the path, keep just
  `force_calc.comp`.
- `SPIRV_OUT` — the target file, e.g. `build/shaders/force_calc.comp.spv`.
- `add_custom_command` — teach the build system a rule:
  - `OUTPUT` — the file this rule produces.
  - `COMMAND` — how: run `glslc force_calc.comp -o build/shaders/force_calc.comp.spv`.
  - `DEPENDS` — rebuild the output if this input changes.
  - `COMMENT` — the line printed during the build.
  - `VERBATIM` — pass the command arguments exactly, no shell mangling.
- `list(APPEND SPIRV_BINARIES ...)` — collect all the `.spv` paths.

```cmake
add_custom_target(shaders ALL DEPENDS ${SPIRV_BINARIES})
```

- A `custom_command` is inert until something *depends* on its output.
- `add_custom_target(shaders ...)` creates a named build target "shaders"
  that depends on every `.spv` file, so building `shaders` compiles them
  all.
- `ALL` — include this target in the default build (so `cmake --build build`
  builds it without being asked).

### The main executable

```cmake
add_executable(profiler
    src/main.cpp
    src/vk_context.cpp
    src/simulation.cpp
    src/report.cpp
)
add_dependencies(profiler shaders)
target_link_libraries(profiler PRIVATE Vulkan::Vulkan)
target_compile_definitions(profiler PRIVATE SHADER_DIR="${SHADER_OUT_DIR}")
```

- `add_executable(profiler ...4 cpp files...)` — the binary `profiler` is
  those four translation units linked together. (Note: `report.cpp` also
  needs `simulation.h`, but headers aren't listed here — only `.cpp` files
  that must be compiled.)
- `add_dependencies(profiler shaders)` — build the shaders *before* the
  executable (so the `.spv` files exist when `profiler` runs).
- `target_link_libraries(profiler PRIVATE Vulkan::Vulkan)` — link the Vulkan
  library into `profiler`. `PRIVATE` = "profiler needs Vulkan, but anything
  linking against profiler doesn't" (irrelevant for an executable, but the
  correct keyword).
- `target_compile_definitions(profiler PRIVATE SHADER_DIR="...")` — this is
  the bridge between build and code. It makes the compiler act as if every
  `.cpp` in `profiler` had `#define SHADER_DIR "/abs/path/to/build/shaders"`
  at the top. `simulation.cpp` uses it:
  `loadShaderModule(ctx.device, std::string(SHADER_DIR) + "/" + file);`
  so the program can find its compiled shaders at runtime regardless of the
  working directory.

### The layer (Phase 9)

```cmake
add_library(VkLayer_gcp_profiler SHARED layer/gcp_layer.cpp)
set_target_properties(VkLayer_gcp_profiler PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    PREFIX "lib")
target_include_directories(VkLayer_gcp_profiler PRIVATE ${Vulkan_INCLUDE_DIRS})
```

- `add_library(name SHARED src)` — build a **shared library** (`.dylib` on
  macOS, `.so` on Linux, `.dll` on Windows), not an executable. A Vulkan
  layer is a shared library the loader `dlopen`s at runtime.
- `CXX_VISIBILITY_PRESET hidden` — by default, hide all symbols
  (functions/variables) in the library. The layer then explicitly *un*-hides
  only the three functions the loader needs (`vkGetInstanceProcAddr`,
  `vkGetDeviceProcAddr`, `vkNegotiateLoaderLayerInterfaceVersion`) via the
  `GCP_EXPORT` attribute in the source. Keeps the library's surface minimal
  and avoids symbol clashes.
- `PREFIX "lib"` — force the output filename to start with `lib`
  (`libVkLayer_gcp_profiler.dylib`). (CMake would add `lib` on Linux anyway;
  this makes it consistent.)
- `target_include_directories(... ${Vulkan_INCLUDE_DIRS})` — the layer needs
  the Vulkan **headers** but must **not link `libvulkan`** (a layer sits
  *inside* the loader; linking the loader would be circular). So we add the
  header path but not `Vulkan::Vulkan`.

### Generating the layer manifest

```cmake
set(GCP_LAYER_LIB "$<TARGET_FILE:VkLayer_gcp_profiler>")
configure_file(${CMAKE_SOURCE_DIR}/layer/VkLayer_gcp_profiler.json.in
               ${CMAKE_BINARY_DIR}/VkLayer_gcp_profiler.json.genin @ONLY)
file(GENERATE
    OUTPUT ${CMAKE_BINARY_DIR}/VkLayer_gcp_profiler.json
    INPUT  ${CMAKE_BINARY_DIR}/VkLayer_gcp_profiler.json.genin)
```

The loader finds a layer via a **JSON manifest** that contains, among other
things, `"library_path"` — where the `.dylib` actually is. We want that path
filled in automatically.

- `layer/VkLayer_gcp_profiler.json.in` is a template with the placeholder
  `@GCP_LAYER_LIB@`.
- `$<TARGET_FILE:VkLayer_gcp_profiler>` is a **generator expression** — a
  value CMake only knows at *build* time (the final path of the built
  library), not configure time.
- Two steps are needed because `configure_file` expands `@VAR@` placeholders
  but *doesn't* understand generator expressions, while `file(GENERATE)`
  expands generator expressions but not `@VAR@`. So:
  1. `configure_file` turns `@GCP_LAYER_LIB@` into the literal generator
     expression `$<TARGET_FILE:...>`, writing a `.genin` intermediate.
  2. `file(GENERATE)` resolves that generator expression to the real path,
     writing the final `VkLayer_gcp_profiler.json`.

Result: `build/VkLayer_gcp_profiler.json` with an absolute
`"library_path"` pointing at `build/libVkLayer_gcp_profiler.dylib`. The
`run_layer.sh` script points the loader at `build/` so it finds this
manifest.

---

## 3. What actually happens when you build

```
$ cmake -S . -B build
  -> reads CMakeLists.txt
  -> finds the C++ compiler, Vulkan, glslc
  -> writes Makefiles + the generated JSON into build/

$ cmake --build build
  -> glslc force_calc.comp  -> build/shaders/force_calc.comp.spv
  -> glslc integrate.comp   -> build/shaders/integrate.comp.spv
  -> glslc stub.comp        -> build/shaders/stub.comp.spv
  -> c++ -c src/main.cpp        -> main.cpp.o
  -> c++ -c src/vk_context.cpp  -> vk_context.cpp.o
  -> c++ -c src/simulation.cpp  -> simulation.cpp.o
  -> c++ -c src/report.cpp      -> report.cpp.o
  -> link the four .o + libvulkan -> build/profiler
  -> c++ -c layer/gcp_layer.cpp -> gcp_layer.cpp.o
  -> link -> build/libVkLayer_gcp_profiler.dylib
```

Incremental: change one `.cpp`, only that file recompiles + a relink. Change
a shader, only that `.spv` rebuilds.

---

## 4. `scripts/run.sh` — the runtime shim

Not part of the build, but part of "how you run it." On macOS the Vulkan
loader can't find the Homebrew-installed validation layer's `.dylib` unless
`DYLD_LIBRARY_PATH` includes the Homebrew `lib` folder. `run.sh` just:

```bash
export DYLD_LIBRARY_PATH="$(brew --prefix)/lib:$DYLD_LIBRARY_PATH"
exec ./build/profiler "$@"
```

(`"$@"` forwards all the arguments you passed to the script.) On Linux with
the SDK on the default path, you can run `./build/profiler` directly.

---

## Explain it in one sentence

> `CMakeLists.txt` finds Vulkan and `glslc`, adds a custom command per
> `.comp` shader to compile it to SPIR-V, builds the `profiler` executable
> from four `.cpp` files while passing the compiled-shader directory in as a
> `-DSHADER_DIR=` define, and separately builds the layer as a
> headers-only-linked shared library with an auto-generated JSON manifest.

## Questions you should be able to answer

1. What do `cmake -S . -B build` and `cmake --build build` each do?
2. How does a `.comp` file become a `.spv` file in this build, and how does
   the running program know where to find it?
3. Why does `add_custom_target(shaders ALL DEPENDS ...)` exist rather than
   just the custom commands?
4. Why is the layer built with `add_library(... SHARED)` and *not* linked
   against `Vulkan::Vulkan`?
5. Why does the layer manifest need a two-step
   `configure_file` + `file(GENERATE)` dance?
6. What does `CXX_VISIBILITY_PRESET hidden` do and why does a layer want it?
