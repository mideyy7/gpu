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
)
add_dependencies(profiler shaders)
target_link_libraries(profiler PRIVATE Vulkan::Vulkan)
target_compile_definitions(profiler PRIVATE SHADER_DIR="${SHADER_OUT_DIR}")
