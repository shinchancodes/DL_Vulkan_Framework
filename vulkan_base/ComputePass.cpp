#include "ComputePass.h"
#include "ComputePipeline.h"
#include <stdexcept>

void ComputePass::record(
    VkDevice device,
    VkCommandPool pool,
    const ComputePipeline& pipeline,
    DispatchSize dispatch,
    const void* pushConstants,
    uint32_t pushConstantSize)
{
    // Allocate command buffer
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool        = pool;
    allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

    // Create fence for waiting
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vkCreateFence(device, &fenceInfo, nullptr, &fence);

    // Record
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline.pipelineLayout, 0, 1, &pipeline.descriptorSet, 0, nullptr);

    if (pushConstants && pushConstantSize > 0) {
        vkCmdPushConstants(commandBuffer, pipeline.pipelineLayout,
            VK_SHADER_STAGE_COMPUTE_BIT, 0, pushConstantSize, pushConstants);
    }

    vkCmdDispatch(commandBuffer, dispatch.x, dispatch.y, dispatch.z);

    vkEndCommandBuffer(commandBuffer);
}

// Implementation:
void ComputePass::submit(VkDevice device, VkQueue queue, bool waitForFence) {
    VkSubmitInfo submitInfo{};
    submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &commandBuffer;

    vkQueueSubmit(queue, 1, &submitInfo, fence);

    if (waitForFence)
        vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
}

void ComputePass::destroy(VkDevice device, VkCommandPool pool) {
    if (fence)         vkDestroyFence(device, fence, nullptr);
    if (commandBuffer) vkFreeCommandBuffers(device, pool, 1, &commandBuffer);
}