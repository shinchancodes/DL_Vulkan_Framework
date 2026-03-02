# ComputeHeadless — Vulkan GPU Compute

A modular, headless Vulkan compute application for running GPU compute shaders
without a display window. Based on Sascha Willems' `computeheadless` example,
refactored into clean, reusable modules.

## Features

- Headless Vulkan compute (no window/swapchain needed)
- Modular architecture — easy to swap shaders and buffers
- Push constant support for passing parameters to shaders
- Host-visible buffer upload/download
- Validation layer support (when available)

## Project Structure

​```
.
├── main.cpp                      # Entry point
├── vulkan_base
|   ├── VulkanContext.h/.cpp      # Instance, device, queue, command pool
|   ├── VulkanBuffer.h/.cpp       # Buffer creation, upload, download
|   ├── ComputePipeline.h/.cpp    # Shader loading, descriptors, pipeline
|   └── ComputePass.h/.cpp        # Command recording and submission
├── shaders/
│   ├── headless.comp             # GLSL compute shader source
│   └── headless.comp.spv         # Compiled SPIR-V (auto-generated)
└── CMakeLists.txt
​```

## Dependencies

| Dependency        | Version | Purpose                        |
|-------------------|---------|--------------------------------|
| Vulkan SDK        | ≥ 1.2   | Headers, loader, layers        |
| CMake             | ≥ 3.16  | Build system                   |
| glslangValidator  | any     | Compile GLSL → SPIR-V          |
| GCC/Clang/MSVC    | C++17   | Compiler                       |

### Install on Ubuntu / WSL2

​```bash
sudo apt update && sudo apt install -y \
    libvulkan1 libvulkan-dev \
    vulkan-tools vulkan-validationlayers \
    mesa-vulkan-drivers \
    glslang-tools cmake build-essential
​```

Full Vulkan SDK: https://vulkan.lunarg.com

## Build

​```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
​```

## Compile shader
'''
glslangValidator -V shaders/headless.comp -o shaders/headless.comp.spv
'''

## Run

​```bash
./build/ComputeHeadless
​```

Expected output (default shader doubles each element):
​```
out[0] = 2
out[1] = 4
out[2] = 6
out[3] = 8
​```

## WSL2 Notes

WSL1 does NOT support Vulkan. On WSL2:

​```bash
ls /dev/dxg              # should exist if GPU passthrough is enabled
vulkaninfo --summary     # should list your GPU
​```

## Extending — Matrix-Vector Multiply

To implement `y = A * x`:
1. Add a third `VulkanBuffer` for the matrix
2. Pass 3 bindings to `ComputePipeline`
3. Add push constants for M and N dimensions
4. Write a new `.comp` shader

No changes to `VulkanContext`, `VulkanBuffer`, or `ComputePass` needed.

## Module Overview

| Module            | Responsibility                                        |
|-------------------|-------------------------------------------------------|
| `VulkanContext`   | Instance, device selection, queue, command pool       |
| `VulkanBuffer`    | GPU buffer allocation, upload/download                |
| `ComputePipeline` | SPIR-V loading, descriptor sets, pipeline creation    |
| `ComputePass`     | Command recording, submission, fence synchronization  |

## License

MIT