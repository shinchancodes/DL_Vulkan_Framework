#include "linear.h"

// vulkan_base headers are one level up
#include "../vulkan_base/VulkanContext.h"
#include "../vulkan_base/VulkanBuffer.h"
#include "../vulkan_base/ComputePipeline.h"
#include "../vulkan_base/ComputePass.h"

#include <stdexcept>
#include <cstring>

struct LinearLayerPushConstants {
    uint32_t M;
    uint32_t K;
};

void Linear::setWeights(const std::vector<float>& weights,
                const std::vector<float>& bias) {
    weights_ = weights;
    
    if (!bias.empty()) {
        if (bias.size() != (size_t)outFeatures_)
            throw std::invalid_argument("Linear Layer: bias size mismatch");
        bias_ = bias;
    } else {
        bias_ = std::vector<float>(outFeatures_, 0.0f); // Default bias = 0
    }
    
}


std::vector<float> Linear::run(const std::vector<float>& x)
{
    if (weights_.size() == 0)
        throw std::invalid_argument("Linear Layer: weights not set");
    uint32_t M = outFeatures_;
    uint32_t K = inFeatures_;
    
    std::vector<float> A = weights_; // Assume weights_ is already in row-major order
    if (A.size() != (size_t)M * K)
        throw std::invalid_argument("Linear Layer: A size mismatch");
    if (x.size() != (size_t)K)
        throw std::invalid_argument("Linear Layer: x size mismatch");

    VkDeviceSize sizeA = A.size() * sizeof(float);
    VkDeviceSize sizeX = x.size() * sizeof(float);
    VkDeviceSize sizeBias = bias_.size() * sizeof(float);
    VkDeviceSize sizeY = (size_t)M * sizeof(float);

    // Create buffers for A, x, y
    VulkanBuffer bufA, bufX, bufBias, bufY;

    bufA.create(ctx_, sizeA, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    bufX.create(ctx_, sizeX, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    bufBias.create(ctx_, sizeBias, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT); 
    bufY.create(ctx_, sizeY, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    bufA.upload(ctx_, A.data(), sizeA);
    bufX.upload(ctx_, x.data(), sizeX);
    bufBias.upload(ctx_, bias_.data(), sizeBias);

    std::vector<DescriptorBinding> buffers = {
        { 0, bufA.buffer, sizeA },
        { 1, bufX.buffer, sizeX },
        { 2, bufBias.buffer, sizeBias },
        { 3, bufY.buffer, sizeY }
    };

    ComputePipeline pipeline;
    pipeline.create(ctx_.device, spvPath_, buffers,
        (uint32_t)sizeof(LinearLayerPushConstants));

    // One workgroup row per thread; dispatch ceil(M / WG) groups
    const uint32_t WG = 64;
    LinearLayerPushConstants pc{ M, K };

    ComputePass pass;
    pass.record(ctx_.device, ctx_.commandPool, pipeline,
                { (M + WG - 1) / WG, 1, 1 },
                &pc, sizeof(LinearLayerPushConstants));
    pass.submit(ctx_.device, ctx_.computeQueue);

    std::vector<float> y(M);
    bufY.download(ctx_, y.data(), sizeY);

    bufA.destroy(ctx_.device);
    bufX.destroy(ctx_.device);
    bufBias.destroy(ctx_.device);
    bufY.destroy(ctx_.device);
    pipeline.destroy(ctx_.device);
    pass.destroy(ctx_.device, ctx_.commandPool);
    return y;
}