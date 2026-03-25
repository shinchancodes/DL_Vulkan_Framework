#pragma once

#include "../vulkan_base/VulkanContext.h"
#include "../vulkan_base/VulkanBuffer.h"
#include "../vulkan_base/ComputePipeline.h"
#include "../vulkan_base/ComputePass.h"

#include <vector>
#include <string>

class Relu {
public:
    Relu(VulkanContext& ctx, const int numElements):
        ctx_(ctx), numElements_(numElements) {};
    ~Relu() = default;

    // Run ReLU on the GPU.
    // X is a host vector in row-major order.
    // Returns Y as a host vector.
    std::vector<float> run(const std::vector<float>& X);
private:
    VulkanContext& ctx_;
    int numElements_;
    std::string spvPath_ = "shaders/spv/relu.spv";
};