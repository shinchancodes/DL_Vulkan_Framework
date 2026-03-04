#include "vulkan_base/VulkanContext.h"
#include "vulkan_base/VulkanBuffer.h"
#include "vulkan_base/ComputePipeline.h"
#include "vulkan_base/ComputePass.h"

#include "layers/linear.h"

#include <iostream>
#include <vector>
#include <string>

int main() {
    // --- Setup ---
    VulkanContext ctx;
    ctx.init(/*enableValidation=*/true);

    std::vector<float> A = { 1, 2, 3, 4, 5, 6 }; // 2x3
    std::vector<float> x = { 7, 8, 9 }; // 3x1
    std::vector<float> bias = { 0.5, 0.5 }; // 2x1
    uint32_t M = 2, K = 3;
    
    Linear linear(ctx, K, M);
    linear.setWeights(A, bias);
    std::vector<float> C = linear.run(x);
    
    // --- Print result ---
    std::cout << "Result C = A * x:" << std::endl;
    for (uint32_t i = 0; i < M; ++i) {
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;

    // --- Cleanup ---
    ctx.destroy();
}