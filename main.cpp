#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>

#include "vulkan_base/VulkanContext.h"
#include "layers/maxpool.h"

int main() {
    VulkanContext ctx;
    ctx.init();

    // 64×64×16 dummy input (replace with real data)
    std::vector<float> input(64 * 64 * 16, 1.0f);

    MaxPoolPushConsts pp { 64, 64, 16 };
    MaxPool pool(ctx, pp);
    auto output = pool.run(input);  // 32×32×16 floats

    std::cout << "Input  size: " << input.size()  << "\n";  // 65536
    std::cout << "Output size: " << output.size() << "\n";  // 16384
    std::cout << "output[0]:   " << output[0]     << "\n";

    ctx.destroy();
    return 0;
}