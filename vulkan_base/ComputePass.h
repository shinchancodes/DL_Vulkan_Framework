#pragma once
#include <vulkan/vulkan.h>
#include "ComputePipeline.h" 

struct DispatchSize { uint32_t x, y, z; };

struct ComputePass {
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;

    void record(
        VkDevice device,
        VkCommandPool pool,
        const ComputePipeline& pipeline,
        DispatchSize dispatch,
        const void* pushConstants     = nullptr,
        uint32_t pushConstantSize     = 0
    );

    void submit(VkDevice device, VkQueue queue, bool waitForFence = true);
    void destroy(VkDevice device, VkCommandPool pool);

private:
    VkFence fence = VK_NULL_HANDLE;
};