#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <string>

struct DescriptorBinding {
    uint32_t        binding;
    VkBuffer        buffer;
    VkDeviceSize    size;
};

struct ComputePipeline {
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool      descriptorPool      = VK_NULL_HANDLE;
    VkDescriptorSet       descriptorSet       = VK_NULL_HANDLE;
    VkPipelineLayout      pipelineLayout      = VK_NULL_HANDLE;
    VkPipeline            pipeline            = VK_NULL_HANDLE;

    void create(
        VkDevice device,
        const std::string& shaderPath,
        const std::vector<DescriptorBinding>& bindings,
        uint32_t pushConstantSize = 0
    );

    void destroy(VkDevice device);

private:
    VkShaderModule loadShader(VkDevice device, const std::string& path);
    void createDescriptorSetLayout(VkDevice device, uint32_t bindingCount);
    void createDescriptorPool(VkDevice device, uint32_t bindingCount);
    void allocateAndUpdateDescriptors(VkDevice device,
                                      const std::vector<DescriptorBinding>& bindings);
};