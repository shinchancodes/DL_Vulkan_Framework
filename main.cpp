#include "vulkan_base/VulkanContext.h"
#include "vulkan_base/VulkanBuffer.h"
#include "vulkan_base/ComputePipeline.h"
#include "vulkan_base/ComputePass.h"

#include "layers/conv2d.h"

#include "imgHelpers/helpers.h"

#include <iostream>
#include <vector>
#include <algorithm>
#include <stdexcept>

// ==========================================================================
// Kernel helpers (same as original main.cpp)
// ==========================================================================
 
/**
 * Build a normalised box-blur kernel.
 * Every spatial position gets weight 1/(kH*kW); each output channel only
 * mixes its corresponding input channel (ic == oc % inC), so colour channels
 * are blurred independently.
 *
 * Layout: [kH, kW, inC, outC]  (same as your conv2d layer expects)
 */
static std::vector<float> makeBlurKernel(int kH, int kW, int inC, int outC)
{
    std::vector<float> k(kH * kW * inC * outC, 0.0f);
    const float w = 1.0f / static_cast<float>(kH * kW);
    for (int ky = 0; ky < kH; ++ky)
        for (int kx = 0; kx < kW; ++kx)
            for (int oc = 0; oc < outC; ++oc)
                for (int ic = 0; ic < inC; ++ic)
                    if (ic == oc % inC)
                        k[((ky * kW + kx) * inC + ic) * outC + oc] = w;
    return k;
}


int main()
{
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
        // 2. Configure convolution
        //    OUT_C must be a multiple you choose; here we keep outC == inC (3)
        //    for a passthrough that can be saved directly.  Change as needed.
        // ------------------------------------------------------------------
        constexpr int OUT_C    = 3;   // set to 8 etc. if you want more feature maps
        constexpr int K_H      = 3;
        constexpr int K_W      = 3;
        constexpr int PAD_H    = 1;
        constexpr int PAD_W    = 1;
        constexpr int STRIDE_H = 1;
        constexpr int STRIDE_W = 1;
 
        ConvPushConsts pc{};
        pc.imgW    = imgW;
        pc.imgH    = imgH;
        pc.inC     = imgC;
        pc.outC    = OUT_C;
        pc.kW      = K_W;
        pc.kH      = K_H;
        pc.padW    = PAD_W;
        pc.padH    = PAD_H;
        pc.strideW = STRIDE_W;
        pc.strideH = STRIDE_H;
 
        // ------------------------------------------------------------------
        // 3. Initialise Vulkan & run the layer
        // ------------------------------------------------------------------
        VulkanContext ctx;
        ctx.init(/*enableValidation=*/true);
 
        Conv2D conv(ctx, pc);
        conv.setWeights(makeBlurKernel(K_H, K_W, imgC, OUT_C));
 
        std::vector<float> hostOutput = conv.run(hostInput);
 
        std::cout << "Conv2D done.  Output elements: " << hostOutput.size() << "\n";
 
        // ------------------------------------------------------------------
        // 4. Compute output spatial dimensions (for strided / padded convs)
        // ------------------------------------------------------------------
        const int outH = (imgH + 2 * PAD_H - K_H) / STRIDE_H + 1;
        const int outW = (imgW + 2 * PAD_W - K_W) / STRIDE_W + 1;
 
        // ------------------------------------------------------------------
        // 5. Save result
        // ------------------------------------------------------------------
        saveImagePNG(outputPath, hostOutput, outH, outW, OUT_C);
        std::cout << "Saved:   " << outputPath
                  << "  [" << outH << "x" << outW << "x" << OUT_C << "]\n";
 
        ctx.destroy();
    }
    catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}