#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <string>

struct VulkanContext {
    VkInstance instance               = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice   = VK_NULL_HANDLE;
    VkDevice device                   = VK_NULL_HANDLE;
    VkQueue computeQueue              = VK_NULL_HANDLE;
    VkCommandPool commandPool         = VK_NULL_HANDLE;
    uint32_t computeQueueFamilyIndex  = 0;

    void init(bool enableValidation = true);
    void destroy();

    uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props) const;

private:
    void createInstance(bool enableValidation);
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createCommandPool();
};