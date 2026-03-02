#include "vulkan_base/VulkanContext.h"
#include "vulkan_base/VulkanBuffer.h"
#include "vulkan_base/ComputePipeline.h"
#include "vulkan_base/ComputePass.h"

#include <iostream>

int main() {
    // --- Setup ---
    VulkanContext ctx;
    ctx.init(/*enableValidation=*/true);

    // --- Data ---
    std::vector<float> inputData  = { 1, 2, 3, 4 };
    std::vector<float> outputData(4, 0.0f);
    VkDeviceSize bufSize = 4 * sizeof(float);

    VulkanBuffer inputBuf, outputBuf;
    inputBuf.create(ctx, bufSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    inputBuf.upload(ctx, inputData.data(), bufSize);

    outputBuf.create(ctx, bufSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // --- Pipeline ---
    ComputePipeline pipeline;
    pipeline.create(ctx.device, "shaders/headless.comp.spv", {
        {0, inputBuf.buffer,  bufSize},
        {1, outputBuf.buffer, bufSize},
    });

    // --- Dispatch ---
    ComputePass pass;
    pass.record(ctx.device, ctx.commandPool, pipeline, {4, 1, 1});
    pass.submit(ctx.device, ctx.computeQueue);

    // --- Read back ---
    outputBuf.download(ctx, outputData.data(), bufSize);
    for (int i = 0; i < 4; i++)
        std::cout << "out[" << i << "] = " << outputData[i] << "\n";

    // --- Cleanup ---
    pass.destroy(ctx.device, ctx.commandPool);
    pipeline.destroy(ctx.device);
    inputBuf.destroy(ctx.device);
    outputBuf.destroy(ctx.device);
    ctx.destroy();
}