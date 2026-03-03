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
//  A : M x K  (row-major, float32)
//  B : K x N  (row-major, float32)
//  C : M x N  (row-major, float32)  — written by the shader
// ---------------------------------------------------------------------------
class MatMul {
public:
    MatMul(VulkanContext& ctx, const std::string& spvPath);
    ~MatMul() = default;

    // Run C = A * B on the GPU.
    // A and B are host vectors in row-major order.
    // Returns C as a host vector.
    std::vector<float> run(const std::vector<float>& A,
                           const std::vector<float>& B,
                           uint32_t M, uint32_t K, uint32_t N);

private:
    VulkanContext& ctx_;
    std::string    spvPath_;
};