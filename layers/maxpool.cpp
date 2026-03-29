// MaxPool.cpp
#include "maxpool.h"

#include <stdexcept>

MaxPool::MaxPool(VulkanContext& ctx, const MaxPoolPushConsts& pushConsts)
    : ctx_(ctx), pushConsts_(pushConsts)
{
    VkDeviceSize sizeInput  = (size_t)(pushConsts_.imgW * pushConsts_.imgH * pushConsts_.channels) * sizeof(float);
    VkDeviceSize sizeOutput = (size_t)((pushConsts_.imgW / 2) * (pushConsts_.imgH / 2) * pushConsts_.channels) * sizeof(float);

    inputBuffer_.create(ctx_, sizeInput,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    outputBuffer_.create(ctx_, sizeOutput,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    std::vector<DescriptorBinding> buffers = {
        { 0, inputBuffer_.buffer,  sizeInput  },
        { 1, outputBuffer_.buffer, sizeOutput }
    };

    pipeline_.create(ctx_.device, spvPath_, buffers,
        (uint32_t)sizeof(MaxPoolPushConsts));

    auto divUp = [](int a, int b){ return (a + b - 1) / b; };
    uint32_t gx = divUp(pushConsts_.imgW / 2, 8);
    uint32_t gy = divUp(pushConsts_.imgH / 2, 8);
    uint32_t gz = static_cast<uint32_t>(pushConsts_.channels);

    pass_.record(ctx_.device, ctx_.commandPool, pipeline_,
                { gx, gy, gz },
                &pushConsts_, sizeof(MaxPoolPushConsts));
}

std::vector<float> MaxPool::run(const std::vector<float>& input) {
    if (input.size() != (size_t)(pushConsts_.imgW * pushConsts_.imgH * pushConsts_.channels))
        throw std::invalid_argument("MaxPool: input size mismatch");

    inputBuffer_.upload(ctx_, input.data(), input.size() * sizeof(float));
    pass_.submit(ctx_.device, ctx_.computeQueue);

    int outW = pushConsts_.imgW / 2;
    int outH = pushConsts_.imgH / 2;
    std::vector<float> output(outW * outH * pushConsts_.channels);
    outputBuffer_.download(ctx_, output.data(), output.size() * sizeof(float));

    inputBuffer_.destroy(ctx_.device);
    outputBuffer_.destroy(ctx_.device);
    pipeline_.destroy(ctx_.device);
    pass_.destroy(ctx_.device, ctx_.commandPool);

    return output;
}