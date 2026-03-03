cmake_minimum_required(VERSION 3.16)
project(VulkanDL_framework)

set(CMAKE_CXX_STANDARD 17)

find_package(Vulkan REQUIRED)

add_executable(VulkanDL_framework
    main.cpp
    vulkan_base/VulkanContext.cpp
    vulkan_base/VulkanBuffer.cpp
    vulkan_base/ComputePipeline.cpp
    vulkan_base/ComputePass.cpp
    operations/Matmul.cpp
)

target_include_directories(VulkanDL_framework PRIVATE ${Vulkan_INCLUDE_DIRS})
target_link_libraries(VulkanDL_framework PRIVATE Vulkan::Vulkan)


find_program(GLSLANG glslangValidator REQUIRED)