// conv2d.cpp
#include "conv2d.h"

#include <stdexcept>
#include <iostream>

Conv2D::Conv2D(VulkanContext& ctx, const ConvPushConsts& pushConsts)
    : ctx_(ctx), pushConsts_(pushConsts)
{
    VkDeviceSize sizeInput  = (size_t)(pushConsts_.imgW * pushConsts_.imgH * pushConsts_.inC) * sizeof(float);
    VkDeviceSize sizeKernel = (size_t)(pushConsts_.kW * pushConsts_.kH * pushConsts_.inC * pushConsts_.outC) * sizeof(float);
    VkDeviceSize sizeBias   = (size_t)(pushConsts_.outC) * sizeof(float);

    int outW = outDim(pushConsts_.imgW, pushConsts_.padW, pushConsts_.kW, pushConsts_.strideW);
    int outH = outDim(pushConsts_.imgH, pushConsts_.padH, pushConsts_.kH, pushConsts_.strideH);
    VkDeviceSize sizeOutput = (size_t)(outW * outH * pushConsts_.outC) * sizeof(float);

    inputBuffer_.create(ctx_, sizeInput,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    kernelBuffer_.create(ctx_, sizeKernel,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    biasBuffer_.create(ctx_, sizeBias,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    outputBuffer_.create(ctx_, sizeOutput,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    std::vector<DescriptorBinding> buffers = {
        { 0, inputBuffer_.buffer,  sizeInput  },
        { 1, kernelBuffer_.buffer, sizeKernel },
        { 2, biasBuffer_.buffer,   sizeBias   },
        { 3, outputBuffer_.buffer, sizeOutput }
    };

    pipeline_.create(ctx_.device, spvPath_, buffers,
        (uint32_t)sizeof(ConvPushConsts));

    auto divUp = [](int a, int b){ return (a + b - 1) / b; };
    uint32_t gx = divUp(outW, 16);
    uint32_t gy = divUp(outH, 16);
    uint32_t gz = static_cast<uint32_t>(pushConsts_.outC);

    pass_.record(ctx_.device, ctx_.commandPool, pipeline_,
                { gx, gy, gz },
                &pushConsts_, sizeof(ConvPushConsts));
}

void Conv2D::setWeights(const std::vector<float>& kernel) {
    if (kernel.size() != (size_t)(pushConsts_.kW * pushConsts_.kH * pushConsts_.inC * pushConsts_.outC))
        throw std::invalid_argument("Conv2D: kernel size mismatch");
    kernelBuffer_.upload(ctx_, kernel.data(), kernel.size() * sizeof(float));
}

void Conv2D::setBias(const std::vector<float>& bias) {
    if (bias.size() != (size_t)(pushConsts_.outC))
        throw std::invalid_argument("Conv2D: bias size mismatch — expected " +
            std::to_string(pushConsts_.outC) + " got " + std::to_string(bias.size()));
    biasBuffer_.upload(ctx_, bias.data(), bias.size() * sizeof(float));
}

std::vector<float> Conv2D::run(const std::vector<float>& input) {
    if (input.size() != (size_t)(pushConsts_.imgW * pushConsts_.imgH * pushConsts_.inC))
        throw std::invalid_argument("Conv2D: input size mismatch");

    inputBuffer_.upload(ctx_, input.data(), input.size() * sizeof(float));
    pass_.submit(ctx_.device, ctx_.computeQueue);

    int outW = outDim(pushConsts_.imgW, pushConsts_.padW, pushConsts_.kW, pushConsts_.strideW);
    int outH = outDim(pushConsts_.imgH, pushConsts_.padH, pushConsts_.kH, pushConsts_.strideH);
    std::vector<float> output(outW * outH * pushConsts_.outC);
    outputBuffer_.download(ctx_, output.data(), output.size() * sizeof(float));

    inputBuffer_.destroy(ctx_.device);
    kernelBuffer_.destroy(ctx_.device);
    biasBuffer_.destroy(ctx_.device);
    outputBuffer_.destroy(ctx_.device);
    pipeline_.destroy(ctx_.device);
    pass_.destroy(ctx_.device, ctx_.commandPool);

    return output;
}