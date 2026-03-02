#include "VulkanContext.h"
#include <stdexcept>
#include <iostream>
#include <vector>
#include <cstring>

void VulkanContext::init(bool enableValidation) {
    createInstance(enableValidation);
    pickPhysicalDevice();
    createLogicalDevice();
    createCommandPool();
}

void VulkanContext::createInstance(bool enableValidation) {
    // First check if validation layer is actually available
    if (enableValidation) {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        bool found = false;
        for (const auto& layer : availableLayers)
            if (strcmp(layer.layerName, "VK_LAYER_KHRONOS_validation") == 0)
                { found = true; break; }

        if (!found) {
            std::cerr << "Validation layer not found, disabling...\n";
            enableValidation = false;
        }
    }

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "ComputeHeadless";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    std::vector<const char*> layers;
    if (enableValidation)
        layers.push_back("VK_LAYER_KHRONOS_validation");

    createInfo.enabledLayerCount   = (uint32_t)layers.size();
    createInfo.ppEnabledLayerNames = layers.data();

    VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);
    if (result != VK_SUCCESS) {
        // Print exact error code
        std::cerr << "vkCreateInstance failed with error: " << result << "\n";
        // Common codes:
        // -9  = VK_ERROR_INCOMPATIBLE_DRIVER
        // -7  = VK_ERROR_EXTENSION_NOT_PRESENT
        // -1  = VK_ERROR_OUT_OF_HOST_MEMORY
        throw std::runtime_error("Failed to create Vulkan instance");
    }
}

void VulkanContext::pickPhysicalDevice() {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    if (count == 0)
        throw std::runtime_error("No Vulkan physical devices found");

    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());
    physicalDevice = devices[0]; // pick first device
}

void VulkanContext::createLogicalDevice() {
    // Find compute queue family
    uint32_t queueCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueCount, nullptr);
    std::vector<VkQueueFamilyProperties> families(queueCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueCount, families.data());

    computeQueueFamilyIndex = UINT32_MAX;
    for (uint32_t i = 0; i < queueCount; i++) {
        if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            computeQueueFamilyIndex = i;
            break;
        }
    }
    if (computeQueueFamilyIndex == UINT32_MAX)
        throw std::runtime_error("No compute queue family found");

    float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{};
    queueInfo.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = computeQueueFamilyIndex;
    queueInfo.queueCount       = 1;
    queueInfo.pQueuePriorities = &priority;

    VkDeviceCreateInfo deviceInfo{};
    deviceInfo.sType            = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos    = &queueInfo;

    if (vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &device) != VK_SUCCESS)
        throw std::runtime_error("Failed to create logical device");

    vkGetDeviceQueue(device, computeQueueFamilyIndex, 0, &computeQueue);
}

void VulkanContext::createCommandPool() {
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = computeQueueFamilyIndex;
    poolInfo.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS)
        throw std::runtime_error("Failed to create command pool");
}

uint32_t VulkanContext::findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props) const {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((typeBits & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    throw std::runtime_error("Failed to find suitable memory type");
}

void VulkanContext::destroy() {
    if (commandPool) vkDestroyCommandPool(device, commandPool, nullptr);
    if (device)      vkDestroyDevice(device, nullptr);
    if (instance)    vkDestroyInstance(instance, nullptr);
}