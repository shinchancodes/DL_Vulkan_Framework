#include "vulkan_base/VulkanContext.h"
#include "vulkan_base/VulkanBuffer.h"
#include "vulkan_base/ComputePipeline.h"
#include "vulkan_base/ComputePass.h"

#include "layers/linear.h"

#include "weightsParser/linear/load_linear_weights.h"
#include "imgHelpers/helpers.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <string>

int main()
{
    std::string weightsDir = "weightsParser/bin";

    const std::string inputPath  = "img/input.jpg";
    const std::string outputPath = "img/output.jpg";

    try {
        // ------------------------------------------------------------------
        // 1. Load SimpleLinear weights from disk
        // ------------------------------------------------------------------
        LinearMeta         meta{};
        std::vector<float> weight, bias;
        loadLinearWeights(weightsDir, meta, weight, bias);

        // ------------------------------------------------------------------
        // 2. Initialise Vulkan & run the layer
        // ------------------------------------------------------------------
        VulkanContext ctx;
        ctx.init(/*enableValidation=*/true);

        Linear linear(ctx, meta.in_features, meta.out_features);
        linear.setWeights(weight, bias);
        
        // Example input — replace with your real data
        std::vector<float> hostInput(meta.in_features, 1.0f);

        std::vector<float> hostOutput = linear.run(hostInput);

        std::cout << "Linear done.  Output elements: "
                  << hostOutput.size() << "\n";
        std::cout << "Output values: ";
        for (float v : hostOutput)
            std::cout << v << " ";
        std::cout << "\n";

        ctx.destroy();
    }

    catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}