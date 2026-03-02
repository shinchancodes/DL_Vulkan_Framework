cmake_minimum_required(VERSION 3.16)
project(ComputeHeadless)

set(CMAKE_CXX_STANDARD 17)

find_package(Vulkan REQUIRED)

add_executable(ComputeHeadless
    main.cpp
    vulkan_base/VulkanContext.cpp
    vulkan_base/VulkanBuffer.cpp
    vulkan_base/ComputePipeline.cpp
    vulkan_base/ComputePass.cpp
)

target_include_directories(ComputeHeadless PRIVATE ${Vulkan_INCLUDE_DIRS})
target_link_libraries(ComputeHeadless PRIVATE Vulkan::Vulkan)


find_program(GLSLANG glslangValidator REQUIRED)

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/shaders/headless.comp.spv
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/shaders
    COMMAND ${GLSLANG} -V
            ${CMAKE_SOURCE_DIR}/shaders/headless.comp
            -o ${CMAKE_BINARY_DIR}/shaders/headless.comp.spv
    DEPENDS shaders/headless.comp
)

add_custom_target(Shaders ALL
    DEPENDS ${CMAKE_BINARY_DIR}/shaders/headless.comp.spv
)

add_dependencies(ComputeHeadless Shaders)