#include "Relu.h"

std::vector<float> Relu::run(const std::vector<float>& X)
{
    VulkanBuffer bufIn, bufOut;
    VkDeviceSize size = X.size() * sizeof(float);

    bufIn.create(ctx_, size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    bufOut.create(ctx_, size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    bufIn.upload(ctx_, X.data(), size);

    std::vector<DescriptorBinding> buffers = {
        { 0, bufIn.buffer, size },
        { 1, bufOut.buffer, size }
    };

    ComputePipeline pipeline;
    pipeline.create(ctx_.device, spvPath_, buffers);
    
    ComputePass pass;
    pass.record(ctx_.device, ctx_.commandPool, pipeline, { (uint32_t)X.size(), 1, 1 });
    pass.submit(ctx_.device, ctx_.computeQueue);
    
    std::vector<float> Y(X.size());
    bufOut.download(ctx_, Y.data(), size);
    return Y;
}