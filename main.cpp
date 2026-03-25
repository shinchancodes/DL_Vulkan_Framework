#include "vulkan_base/VulkanContext.h"

#include "layers/Relu.h"

#include <vector>
#include <stdio.h>

int main()
{
    VulkanContext ctx;
    ctx.init();

    // Example input
    const std::vector<float> X = { -1.0f, 0.0f, 1.0f, 2.0f, -0.5f, 3.0f, -2.0f, 4.0f };
    Relu reluLayer(ctx, (int)X.size());
    std::vector<float> Y = reluLayer.run(X);

    // Print the output
    for (size_t i = 0; i < Y.size(); ++i) {
        printf("Y[%zu] = %f\n", i, Y[i]);
    }
    return 0;
}