// MaxPool.h
#pragma once

#include "../vulkan_base/VulkanContext.h"
#include "../vulkan_base/ComputePipeline.h"
#include "../vulkan_base/VulkanBuffer.h"
#include "../vulkan_base/ComputePass.h"

#include <string>

struct MaxPoolPushConsts {
    int32_t imgW;
    int32_t imgH;
    int32_t channels;
};

class MaxPool {
public:
    MaxPool(VulkanContext& ctx, const MaxPoolPushConsts& pushConsts);
    ~MaxPool() {};

    std::vector<float> run(const std::vector<float>& input);

private:
    VulkanContext& ctx_;
    VulkanBuffer inputBuffer_, outputBuffer_;
    ComputePipeline pipeline_;
    ComputePass pass_;

    MaxPoolPushConsts pushConsts_;

    std::string spvPath_ = "shaders/spv/maxpool.spv";
};