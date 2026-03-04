#pragma once

#include "../vulkan_base/VulkanContext.h"
#include "../vulkan_base/VulkanBuffer.h"
#include "../vulkan_base/ComputePipeline.h"
#include "../vulkan_base/ComputePass.h"

#include <vector>
#include <string>

// ---------------------------------------------------------------------------
//  MatMul  —  GPU Matrix Multiplication  C = A * B
//
//  Weights : M x K  (row-major, float32)
//  B : K x 1  (row-major, float32)
//  C : M x 1  (row-major, float32)  — written by the shader
// ---------------------------------------------------------------------------
class Linear {
public:
    Linear(VulkanContext& ctx, const int inFeatures, const int outFeatures):
        ctx_(ctx), inFeatures_(inFeatures), outFeatures_(outFeatures) {};
    ~Linear() = default;

    // Run C = A * X on the GPU.
    // A and X are host vectors in row-major order.
    // Returns C as a host vector.
    std::vector<float> run(const std::vector<float>& X);
    
    void setWeights(const std::vector<float>& weights,
                    const std::vector<float>& bias = {});
                    
private:
    VulkanContext& ctx_;
    int inFeatures_;
    int outFeatures_;
    std::string spvPath_ = "shaders/spv/linear.spv";
    std::vector<float> weights_;
    std::vector<float> bias_;
};