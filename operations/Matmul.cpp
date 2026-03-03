#include "Matmul.h"

// vulkan_base headers are one level up
#include "../vulkan_base/VulkanContext.h"
#include "../vulkan_base/VulkanBuffer.h"
#include "../vulkan_base/ComputePipeline.h"
#include "../vulkan_base/ComputePass.h"

#include <stdexcept>
#include <cstring>

struct MatMulPushConstants {
    uint32_t M;
    uint32_t K;
    uint32_t N;
};

MatMul::MatMul(VulkanContext& ctx, const std::string& spvPath)
    : ctx_(ctx), spvPath_(spvPath) {}

std::vector<float> MatMul::run(const std::vector<float>& A,
                               const std::vector<float>& B,
                               uint32_t M, uint32_t K, uint32_t N)
{
    if (A.size() != (size_t)M * K)
        throw std::invalid_argument("MatMul: A size mismatch");
    if (B.size() != (size_t)K * N)
        throw std::invalid_argument("MatMul: B size mismatch");

    VkDeviceSize sizeA = A.size() * sizeof(float);
    VkDeviceSize sizeB = B.size() * sizeof(float);
    VkDeviceSize sizeC = (size_t)M * N * sizeof(float);

    // Create buffers for A, B, C
    VulkanBuffer bufA, bufB, bufC;
    
    bufA.create(ctx_, sizeA, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    bufB.create(ctx_, sizeB, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    bufC.create(ctx_, sizeC, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    bufA.upload(ctx_, A.data(), sizeA);
    bufB.upload(ctx_, B.data(), sizeB);

    std::vector<DescriptorBinding> buffers = {
        { 0, bufA.buffer, sizeA },
        { 1, bufB.buffer, sizeB },
        { 2, bufC.buffer, sizeC }
    };

    ComputePipeline pipeline;
    pipeline.create(ctx_.device, spvPath_, buffers,
        (uint32_t)sizeof(MatMulPushConstants));

    const uint32_t WG = 16;
    MatMulPushConstants pc{ M, K, N };

    ComputePass pass;
    pass.record(ctx_.device, ctx_.commandPool, pipeline,  { (M+WG-1)/WG, (N+WG-1)/WG, 1 },
                &pc, sizeof(MatMulPushConstants));
    pass.submit(ctx_.device, ctx_.computeQueue);

    std::vector<float> C(M * N);
    bufC.download(ctx_, C.data(), sizeC);

    bufA.destroy(ctx_.device);
    bufB.destroy(ctx_.device);
    bufC.destroy(ctx_.device);
    pipeline.destroy(ctx_.device);
    pass.destroy(ctx_.device, ctx_.commandPool);
    return C;
}