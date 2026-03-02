#pragma once
#include <vulkan/vulkan.h>
#include <cstring>

struct VulkanContext;

struct VulkanBuffer {
    VkBuffer       buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize   size   = 0;

    void create(
        const VulkanContext& ctx,
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        VkMemoryPropertyFlags memProps
    );

    void upload(const VulkanContext& ctx, const void* data, VkDeviceSize dataSize);
    void download(const VulkanContext& ctx, void* data, VkDeviceSize dataSize);
    void destroy(VkDevice device);
};