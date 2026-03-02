#include "ComputePipeline.h"
#include <fstream>
#include <stdexcept>
#include <vector>

static std::vector<char> readFile(const std::string& path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Failed to open shader: " + path);
    size_t size = file.tellg();
    std::vector<char> buf(size);
    file.seekg(0);
    file.read(buf.data(), size);
    return buf;
}

VkShaderModule ComputePipeline::loadShader(VkDevice device, const std::string& path) {
    auto code = readFile(path);
    VkShaderModuleCreateInfo info{};
    info.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = code.size();
    info.pCode    = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule mod;
    if (vkCreateShaderModule(device, &info, nullptr, &mod) != VK_SUCCESS)
        throw std::runtime_error("Failed to create shader module");
    return mod;
}

void ComputePipeline::createDescriptorSetLayout(VkDevice device, uint32_t bindingCount) {
    std::vector<VkDescriptorSetLayoutBinding> bindings(bindingCount);
    for (uint32_t i = 0; i < bindingCount; i++) {
        bindings[i].binding         = i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo info{};
    info.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    info.bindingCount = bindingCount;
    info.pBindings    = bindings.data();
    vkCreateDescriptorSetLayout(device, &info, nullptr, &descriptorSetLayout);
}

void ComputePipeline::createDescriptorPool(VkDevice device, uint32_t bindingCount) {
    VkDescriptorPoolSize poolSize{};
    poolSize.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = bindingCount;

    VkDescriptorPoolCreateInfo info{};
    info.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    info.maxSets       = 1;
    info.poolSizeCount = 1;
    info.pPoolSizes    = &poolSize;
    vkCreateDescriptorPool(device, &info, nullptr, &descriptorPool);
}

void ComputePipeline::allocateAndUpdateDescriptors(
    VkDevice device,
    const std::vector<DescriptorBinding>& bindings)
{
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool     = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts        = &descriptorSetLayout;
    vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet);

    std::vector<VkDescriptorBufferInfo> bufInfos(bindings.size());
    std::vector<VkWriteDescriptorSet>   writes(bindings.size());

    for (size_t i = 0; i < bindings.size(); i++) {
        bufInfos[i].buffer = bindings[i].buffer;
        bufInfos[i].offset = 0;
        bufInfos[i].range  = bindings[i].size;

        writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet          = descriptorSet;
        writes[i].dstBinding      = bindings[i].binding;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &bufInfos[i];
    }
    vkUpdateDescriptorSets(device, (uint32_t)writes.size(), writes.data(), 0, nullptr);
}

void ComputePipeline::create(
    VkDevice device,
    const std::string& shaderPath,
    const std::vector<DescriptorBinding>& bindings,
    uint32_t pushConstantSize)
{
    uint32_t bindingCount = (uint32_t)bindings.size();

    createDescriptorSetLayout(device, bindingCount);
    createDescriptorPool(device, bindingCount);
    allocateAndUpdateDescriptors(device, bindings);

    // Pipeline layout
    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts    = &descriptorSetLayout;

    VkPushConstantRange pushRange{};
    if (pushConstantSize > 0) {
        pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushRange.offset     = 0;
        pushRange.size       = pushConstantSize;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges    = &pushRange;
    }
    vkCreatePipelineLayout(device, &layoutInfo, nullptr, &pipelineLayout);

    // Shader stage
    VkShaderModule shaderModule = loadShader(device, shaderPath);

    VkPipelineShaderStageCreateInfo stageInfo{};
    stageInfo.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = shaderModule;
    stageInfo.pName  = "main";

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage  = stageInfo;
    pipelineInfo.layout = pipelineLayout;

    vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);

    vkDestroyShaderModule(device, shaderModule, nullptr); // no longer needed
}

void ComputePipeline::destroy(VkDevice device) {
    if (pipeline)            vkDestroyPipeline(device, pipeline, nullptr);
    if (pipelineLayout)      vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    if (descriptorPool)      vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    if (descriptorSetLayout) vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
}