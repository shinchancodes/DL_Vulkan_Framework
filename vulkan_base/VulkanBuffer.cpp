#include "VulkanBuffer.h"
#include "VulkanContext.h"
#include <stdexcept>
#include <cstring>

void VulkanBuffer::create(
    const VulkanContext& ctx,
    VkDeviceSize sz,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags memProps)
{
    size = sz;

    VkBufferCreateInfo bufInfo{};
    bufInfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size        = size;
    bufInfo.usage       = usage;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(ctx.device, &bufInfo, nullptr, &buffer) != VK_SUCCESS)
        throw std::runtime_error("Failed to create buffer");

    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(ctx.device, buffer, &memReqs);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize  = memReqs.size;
    allocInfo.memoryTypeIndex = ctx.findMemoryType(memReqs.memoryTypeBits, memProps);

    if (vkAllocateMemory(ctx.device, &allocInfo, nullptr, &memory) != VK_SUCCESS)
        throw std::runtime_error("Failed to allocate buffer memory");

    vkBindBufferMemory(ctx.device, buffer, memory, 0);
}

void VulkanBuffer::upload(const VulkanContext& ctx, const void* data, VkDeviceSize dataSize) {
    void* mapped;
    vkMapMemory(ctx.device, memory, 0, dataSize, 0, &mapped);
    memcpy(mapped, data, dataSize);
    vkUnmapMemory(ctx.device, memory);
}

void VulkanBuffer::download(const VulkanContext& ctx, void* data, VkDeviceSize dataSize) {
    void* mapped;
    vkMapMemory(ctx.device, memory, 0, dataSize, 0, &mapped);
    memcpy(data, mapped, dataSize);
    vkUnmapMemory(ctx.device, memory);
}

void VulkanBuffer::destroy(VkDevice device) {
    if (buffer) vkDestroyBuffer(device, buffer, nullptr);
    if (memory) vkFreeMemory(device, memory, nullptr);
    buffer = VK_NULL_HANDLE;
    memory = VK_NULL_HANDLE;
}