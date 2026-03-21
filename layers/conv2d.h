#pragma once

#include "../vulkan_base/VulkanContext.h"
#include "../vulkan_base/ComputePipeline.h"
#include "../vulkan_base/VulkanBuffer.h"
#include "../vulkan_base/ComputePass.h"

#include <string>

struct ConvPushConsts {
    int32_t imgW;
    int32_t imgH;
    int32_t inC;
    int32_t outC;
    int32_t kW;
    int32_t kH;
    int32_t padW;
    int32_t padH;
    int32_t strideW;
    int32_t strideH;
};

class Conv2D {
public:
    Conv2D(VulkanContext& ctx, const ConvPushConsts& pushConsts);
    ~Conv2D() {};

    void setWeights(const std::vector<float>& kernel);
    std::vector<float> run(const std::vector<float>& input);

private:
    VulkanContext& ctx_;
    VulkanBuffer inputBuffer_, kernelBuffer_, outputBuffer_;
    ComputePipeline pipeline_;
    ComputePass pass_;
    
    ConvPushConsts pushConsts_;

    std::string spvPath_ = "shaders/spv/conv2d.spv";

    static int outDim(int in, int pad, int k, int stride) {
        return (in + 2 * pad - k) / stride + 1;
    }
};