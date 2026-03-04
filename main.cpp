#include "vulkan_base/VulkanContext.h"
#include "vulkan_base/VulkanBuffer.h"
#include "vulkan_base/ComputePipeline.h"
#include "vulkan_base/ComputePass.h"

#include "utils/gemm.h"

#include <iostream>
#include <vector>
#include <string>

int main() {
    // --- Setup ---
    VulkanContext ctx;
    ctx.init(/*enableValidation=*/true);

    std::vector<float> A = { 1, 2, 3, 4, 5, 6 }; // 2x3
    std::vector<float> B = { 7, 8, 9 , 10, 11, 12 }; // 3x2
    uint32_t M = 2, K = 3, N = 2;

    GEMM gemm(ctx);

    std::vector<float> C = gemm.run(A, B, M, K, N);

    // --- Print result ---
    std::cout << "Result C = A * B:" << std::endl;
    for (uint32_t i = 0; i < M; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            std::cout << C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // --- Cleanup ---
    ctx.destroy();
}