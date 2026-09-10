cmake_minimum_required(VERSION 3.20)
project(gpu_compute_profiler CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

find_package(Vulkan REQUIRED)

find_program(GLSLC glslc REQUIRED)

set(SHADER_SRC_DIR ${CMAKE_SOURCE_DIR}/shaders)
set(SHADER_OUT_DIR ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_OUT_DIR})

file(GLOB SHADER_SOURCES CONFIGURE_DEPENDS ${SHADER_SRC_DIR}/*.comp)

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

add_custom_target(shaders ALL DEPENDS ${SPIRV_BINARIES})

add_executable(profiler
    src/main.cpp
    src/vk_context.cpp
    src/simulation.cpp
    src/report.cpp
)
add_dependencies(profiler shaders)
target_link_libraries(profiler PRIVATE Vulkan::Vulkan)
target_compile_definitions(profiler PRIVATE SHADER_DIR="${SHADER_OUT_DIR}")

# --- Reusable profiling layer (Phase 9) --------------------------------------
# A layer sits between the app and the loader, so it links the Vulkan HEADERS
# only, never the loader library.
add_library(VkLayer_gcp_profiler SHARED layer/gcp_layer.cpp)
set_target_properties(VkLayer_gcp_profiler PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    PREFIX "lib")
target_include_directories(VkLayer_gcp_profiler PRIVATE ${Vulkan_INCLUDE_DIRS})

# Two-step manifest generation: configure_file expands @GCP_LAYER_LIB@ into a
# generator expression, then file(GENERATE) resolves that to the real path.
set(GCP_LAYER_LIB "$<TARGET_FILE:VkLayer_gcp_profiler>")
configure_file(${CMAKE_SOURCE_DIR}/layer/VkLayer_gcp_profiler.json.in
               ${CMAKE_BINARY_DIR}/VkLayer_gcp_profiler.json.genin @ONLY)
file(GENERATE
    OUTPUT ${CMAKE_BINARY_DIR}/VkLayer_gcp_profiler.json
    INPUT  ${CMAKE_BINARY_DIR}/VkLayer_gcp_profiler.json.genin)
