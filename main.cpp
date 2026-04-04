#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>
#include <numeric>
#include <algorithm>

#include "vulkan_base/VulkanContext.h"

#include "layers/conv2d.h"
#include "layers/maxpool.h"
#include "layers/Relu.h"
#include "layers/linear.h"

#include "imgHelpers/helpers.h"

#define COMPARE_LAYER
#include <iomanip>

// ── Weight loader ─────────────────────────────────────────────────────────────

std::vector<float> loadBin(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    size_t bytes = f.tellg();
    f.seekg(0);
    std::vector<float> data(bytes / sizeof(float));
    f.read(reinterpret_cast<char*>(data.data()), bytes);
    return data;
}




// ── Softmax ───────────────────────────────────────────────────────────────────

std::vector<float> softmax(const std::vector<float>& logits) {
    float maxVal = *std::max_element(logits.begin(), logits.end());
    std::vector<float> probs(logits.size());
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - maxVal);
        sum += probs[i];
    }
    for (auto& p : probs) p /= sum;
    return probs;
}


#ifdef COMPARE_LAYER
// ── Comparison helper ─────────────────────────────────────────────────────────

void compare(const std::vector<float>& vulkan,
             const std::string& refPath,
             const std::string& layerName)
{
    // Load reference
    std::ifstream f(refPath, std::ios::binary | std::ios::ate);
    if (!f) { std::cerr << "  [SKIP] " << refPath << " not found\n"; return; }
    size_t bytes = f.tellg(); f.seekg(0);
    std::vector<float> ref(bytes / sizeof(float));
    f.read(reinterpret_cast<char*>(ref.data()), bytes);

    if (ref.size() != vulkan.size()) {
        std::cerr << "  [" << layerName << "] SIZE MISMATCH"
                  << "  ref=" << ref.size()
                  << "  vulkan=" << vulkan.size() << "\n";
        return;
    }

    float maxErr = 0.0f, sumErr = 0.0f;
    int   worstIdx = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        float err = std::abs(vulkan[i] - ref[i]);
        sumErr += err;
        if (err > maxErr) { maxErr = err; worstIdx = (int)i; }
    }
    float avgErr = sumErr / ref.size();

    std::cout << "  [" << layerName << "]"
              << "  max_err=" << maxErr
              << "  avg_err=" << avgErr
              << "  worst_idx=" << worstIdx
              << "  ref=" << ref[worstIdx]
              << "  vulkan=" << vulkan[worstIdx]
              << (maxErr < 1e-3f ? "  ✓ OK" : "  ✗ MISMATCH") << "\n";
}

void debugConv1(const std::vector<float>& input,
                const std::vector<float>& weights,
                const std::vector<float>& bias,
                const std::vector<float>& vulkanOut)
{
    // Manually compute output[oy=0, ox=0, oc=0] on CPU
    // Input  layout: [H, W, inC]    → (iy*W + ix)*inC + ic
    // Weight layout: [kH, kW, inC, outC] → ((kh*kW + kw)*inC + ic)*outC + oc
    const int W=64, H=64, inC=3, outC=16, kH=3, kW=3, pad=1;

    int oc = 0;
    float acc = bias[oc];
    for (int ic = 0; ic < inC; ++ic) {
        for (int kh = 0; kh < kH; ++kh) {
            for (int kw = 0; kw < kW; ++kw) {
                int ih = 0 + kh - pad;
                int iw = 0 + kw - pad;
                if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                float in_val = input[(ih * W + iw) * inC + ic];
                float w_val  = weights[((kh * kW + kw) * inC + ic) * outC + oc];
                acc += in_val * w_val;
            }
        }
    }

    std::cout << "\n=== Conv1 Debug [oy=0, ox=0, oc=0] ===\n";
    std::cout << "  CPU manual   = " << acc               << "\n";
    std::cout << "  Vulkan out   = " << vulkanOut[0]      << "\n";   // [0,0,0] is index 0
    std::cout << "  bias[0]      = " << bias[0]           << "\n";
    std::cout << "  input[0,0,0] = " << input[0]          << "\n";
    std::cout << "  input[0,0,1] = " << input[1]          << "\n";
    std::cout << "  input[0,0,2] = " << input[2]          << "\n";
    std::cout << "  weight[0]    = " << weights[0]        << "\n";

    // Check first 5 output values
    std::cout << "\n  First 5 vulkan outputs (all oc at [0,0]):\n";
    for (int i = 0; i < 5; ++i)
        std::cout << "    oc=" << i << "  " << vulkanOut[i] << "\n";
}

void deepCompare(const std::vector<float>& vulkan,
                 const std::string& refPath,
                 const std::string& layerName,
                 int H, int W, int C)
{
    std::ifstream f(refPath, std::ios::binary | std::ios::ate);
    if (!f) { std::cerr << "Cannot open " << refPath << "\n"; return; }
    size_t bytes = f.tellg(); f.seekg(0);
    std::vector<float> ref(bytes / sizeof(float));
    f.read(reinterpret_cast<char*>(ref.data()), bytes);

    std::cout << "\n=== " << layerName << " deep compare ===\n";
    std::cout << "  ref size=" << ref.size() << "  vulkan size=" << vulkan.size() << "\n";

    int mismatches = 0;
    float maxErr = 0.0f;
    int worstIdx = 0;

    for (size_t i = 0; i < std::min(ref.size(), vulkan.size()); ++i) {
        float err = std::abs(ref[i] - vulkan[i]);
        if (err > 1e-3f) {
            ++mismatches;
            if (err > maxErr) { maxErr = err; worstIdx = (int)i; }

            // Print first 10 mismatches with spatial coords
            if (mismatches <= 10) {
                int c  =  i % C;
                int x  = (i / C) % W;
                int y  = (i / C) / W;
                std::cout << "  MISMATCH [y=" << y << " x=" << x << " c=" << c << "]"
                          << "  ref=" << ref[i]
                          << "  vulkan=" << vulkan[i]
                          << "  err=" << err << "\n";
            }
        }
    }

    // Print a small spatial patch around worst mismatch
    int wc =  worstIdx % C;
    int wx = (worstIdx / C) % W;
    int wy = (worstIdx / C) / W;
    std::cout << "  Worst mismatch at [y=" << wy << " x=" << wx << " c=" << wc << "]\n";
    std::cout << "  Total mismatches: " << mismatches
              << " / " << ref.size()
              << "  (" << 100.f * mismatches / ref.size() << "%)\n";

    // Print first 16 values of both side by side
    std::cout << "\n  First 16 values [ref | vulkan]:\n";
    for (int i = 0; i < std::min(16, (int)ref.size()); ++i)
        std::cout << "    [" << i << "]  " << ref[i] << "  |  " << vulkan[i] << "\n";
}


#endif

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    const std::string imagePath  = "img/9828.jpg";
    const std::string weightsDir = "weightsParser/bin";

    // ── 1. Load image ─────────────────────────────────────────────────────────
    std::cout << "Loading image: " << imagePath << "\n";
    std::vector<float> input = loadImage(imagePath);  // [64*64*3]
    std::cout << "  Preprocessed → 64×64×3 = " << input.size() << " floats\n";

    // ── 2. Load weights ───────────────────────────────────────────────────────
    std::cout << "\nLoading weights from: " << weightsDir << "/\n";
    auto conv1_w = loadBin(weightsDir + "/conv1_w.bin");  // [3,3,3,16]
    auto conv1_b = loadBin(weightsDir + "/conv1_b.bin");  // [16]
    auto conv2_w = loadBin(weightsDir + "/conv2_w.bin");  // [3,3,16,32]
    auto conv2_b = loadBin(weightsDir + "/conv2_b.bin");  // [32]
    auto conv3_w = loadBin(weightsDir + "/conv3_w.bin");  // [3,3,32,64]
    auto conv3_b = loadBin(weightsDir + "/conv3_b.bin");  // [64]
    auto fc1_w   = loadBin(weightsDir + "/fc1_w.bin");    // [128, 4096]
    auto fc1_b   = loadBin(weightsDir + "/fc1_b.bin");    // [128]
    auto fc2_w   = loadBin(weightsDir + "/fc2_w.bin");    // [2, 128]
    auto fc2_b   = loadBin(weightsDir + "/fc2_b.bin");    // [2]
    std::cout << "  All weights loaded\n";

    // ── 3. Vulkan init ────────────────────────────────────────────────────────
    std::cout << "\nInitialising Vulkan...\n";
    VulkanContext ctx;
    ctx.init(false);

    // ── 4. Conv1 + ReLU + Pool1 ───────────────────────────────────────────────
    //      64×64×3 → 64×64×16 → 32×32×16
    std::cout << "\nRunning inference...\n";

    ConvPushConsts cp1 { 64, 64, 3, 16, 3, 3, 1, 1, 1, 1 };
    Conv2D conv1(ctx, cp1);
    conv1.setWeights(conv1_w);
    conv1.setBias(conv1_b);
    auto out_conv1 = conv1.run(input);          // [64*64*16]

    #ifdef COMPARE_LAYER
    float minVal = *std::min_element(out_conv1.begin(), out_conv1.end());
    float maxVal = *std::max_element(out_conv1.begin(), out_conv1.end());
    std::cout << "Conv1 Max min: " << maxVal << " " << minVal << std::endl;
    
    std::cout << "out_conv1 Weights: \n";
    std::cout << std::fixed << std::setprecision(4);
    for (int i = 0; i < conv1_w.size(); ++i) {
        std::cout << conv1_w[i] << " ";
        if ((i + 1) % 27 == 0) std::cout << "\n";
    }

    #endif

    Relu relu1(ctx, 64 * 64 * 16);
    auto out_relu1 = relu1.run(out_conv1);      // [64*64*16]
    
    #ifdef COMPARE_LAYER
    compare(out_relu1, "activations/activations/after_relu1.bin", "relu1");
    #endif

    MaxPoolPushConsts pp1 { 64, 64, 16 };
    MaxPool pool1(ctx, pp1);
    auto out_pool1 = pool1.run(out_relu1);      // [32*32*16]

    // ── 5. Conv2 + ReLU + Pool2 ───────────────────────────────────────────────
    //      32×32×16 → 32×32×32 → 16×16×32

    ConvPushConsts cp2 { 32, 32, 16, 32, 3, 3, 1, 1, 1, 1 };
    Conv2D conv2(ctx, cp2);
    conv2.setWeights(conv2_w);
    conv2.setBias(conv2_b);
    auto out_conv2 = conv2.run(out_pool1);      // [32*32*32]

    Relu relu2(ctx, 32 * 32 * 32);
    auto out_relu2 = relu2.run(out_conv2);      // [32*32*32]

    MaxPoolPushConsts pp2 { 32, 32, 32 };
    MaxPool pool2(ctx, pp2);
    auto out_pool2 = pool2.run(out_relu2);      // [16*16*32]

    // ── 6. Conv3 + ReLU + Pool3 ───────────────────────────────────────────────
    //      16×16×32 → 16×16×64 → 8×8×64

    ConvPushConsts cp3 { 16, 16, 32, 64, 3, 3, 1, 1, 1, 1 };
    Conv2D conv3(ctx, cp3);
    conv3.setWeights(conv3_w);
    conv3.setBias(conv3_b);
    auto out_conv3 = conv3.run(out_pool2);      // [16*16*64]

    Relu relu3(ctx, 16 * 16 * 64);
    auto out_relu3 = relu3.run(out_conv3);      // [16*16*64]

    MaxPoolPushConsts pp3 { 16, 16, 64 };
    MaxPool pool3(ctx, pp3);
    auto out_pool3 = pool3.run(out_relu3);      // [8*8*64 = 4096]

    // ── 7. FC1 + ReLU ─────────────────────────────────────────────────────────
    //      4096 → 128

    Linear fc1(ctx, 4096, 128);
    fc1.setWeights(fc1_w, fc1_b);
    auto out_fc1 = fc1.run(out_pool3);          // [128]

    Relu relu4(ctx, 128);
    auto out_relu4 = relu4.run(out_fc1);        // [128]

    // ── 8. FC2 ────────────────────────────────────────────────────────────────
    //      128 → 2

    Linear fc2(ctx, 128, 2);
    fc2.setWeights(fc2_w, fc2_b);
    auto logits = fc2.run(out_relu4);             // [2]

    // ── 9. Softmax + result ───────────────────────────────────────────────────

    auto probs = softmax(logits);

    std::cout << "\n=== CatDogNet Inference ===\n";
    std::cout << "  Logits : cat=" << logits[0] << "  dog=" << logits[1] << "\n";
    std::cout << "  Probs  : cat=" << probs[0] * 100.f << "%"
              <<            "  dog=" << probs[1] * 100.f << "%\n";
    std::cout << "  Result : >> " << (probs[0] > probs[1] ? "CAT" : "DOG") << " <<\n";

    // ── 10. Cleanup ───────────────────────────────────────────────────────────
    ctx.destroy();

    return 0;
}