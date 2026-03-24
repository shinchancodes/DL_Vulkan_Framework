#include "vulkan_base/VulkanContext.h"
#include "vulkan_base/VulkanBuffer.h"
#include "vulkan_base/ComputePipeline.h"
#include "vulkan_base/ComputePass.h"

#include "layers/conv2d.h"

#include "preprocess/loadWeights.h"
#include "imgHelpers/helpers.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <string>


int main()
{
    std::string metaPath   = "preprocess/bin/meta.json";
    std::string kernelPath = "preprocess/bin/kernel.bin";

    const std::string inputPath  = "img/input.jpg";
    const std::string outputPath = "img/output.jpg";

    try {
        // ------------------------------------------------------------------
        // 1. Load image
        // ------------------------------------------------------------------
        int imgH, imgW, imgC;
        // forceC=3 → always decode as RGB (drop alpha if present)
        std::vector<float> hostInput = loadImage(inputPath, imgH, imgW, imgC, /*forceC=*/3);
        imgC = 3;  // forceC ensures this
 
        std::cout << "Loaded:  " << inputPath
                  << "  [" << imgH << "x" << imgW << "x" << imgC << "]\n";
 
        // ------------------------------------------------------------------
        // 2. Load SimpleCNN conv1 weights
        // ------------------------------------------------------------------
        KernelMeta km{};
        std::vector<float> kernelWeights =
            loadKernelFromFiles(metaPath, kernelPath, km);

        if (km.inC != imgC)
            throw std::runtime_error(
                "Kernel inC=" + std::to_string(km.inC) +
                " does not match image channels=" + std::to_string(imgC));

        // ------------------------------------------------------------------
        // 3. Build push constants directly from meta — no hardcoding
        // ------------------------------------------------------------------
        ConvPushConsts pc{};
        pc.imgW    = imgW;
        pc.imgH    = imgH;
        pc.inC     = km.inC;
        pc.outC    = km.outC;
        pc.kW      = km.kW;
        pc.kH      = km.kH;
        pc.padW    = km.padW;
        pc.padH    = km.padH;
        pc.strideW = km.strideW;
        pc.strideH = km.strideH;
 
        // ------------------------------------------------------------------
        // 4. Initialise Vulkan & run the layer
        // ------------------------------------------------------------------
        VulkanContext ctx;
        ctx.init(/*enableValidation=*/true);
 
        Conv2D conv(ctx, pc);
        conv.setWeights(kernelWeights);
 
        std::vector<float> hostOutput = conv.run(hostInput);
 
        std::cout << "Conv2D done.  Output elements: " << hostOutput.size() << "\n";
 
        // ------------------------------------------------------------------
        // 5. Compute output spatial dimensions (for strided / padded convs)
        // ------------------------------------------------------------------
        const int outH = (imgH + 2 * km.padH - km.kH) / km.strideH + 1;
        const int outW = (imgW + 2 * km.padW - km.kW) / km.strideW + 1;
 
        // ------------------------------------------------------------------
        // 6. Save result
        // ------------------------------------------------------------------
        saveImagePNG(outputPath, hostOutput, outH, outW, km.outC);
        std::cout << "Saved:   " << outputPath
                  << "  [" << outH << "x" << outW << "x" << km.outC << "]\n";
 
        ctx.destroy();
    }
    catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}