#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>
#include <numeric>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#include "imgHelpers/stb/stb_image.h"

#include "vulkan_base/VulkanContext.h"

#include "layers/conv2d.h"
#include "layers/maxpool.h"
#include "layers/Relu.h"
#include "layers/linear.h"

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

// ── Image loader & preprocessor ───────────────────────────────────────────────
//
//   Loads any image via stb_image, resizes to 64×64, converts to float,
//   normalizes to [-1, 1] and returns in NHWC layout [64*64*3].
//   Matches the Python preprocess(): arr = (arr / 255 - 0.5) / 0.5

std::vector<float> loadImage(const std::string& path) {
    int w, h, c;
    unsigned char* px = stbi_load(path.c_str(), &w, &h, &c, 3);
    if (!px) throw std::runtime_error("Cannot load image: " + path);

    // Manual bilinear resize to 64×64
    const int OUT = 64;
    std::vector<float> img(OUT * OUT * 3);
    float scaleX = (float)w / OUT;
    float scaleY = (float)h / OUT;

    for (int oy = 0; oy < OUT; ++oy) {
        for (int ox = 0; ox < OUT; ++ox) {
            int ix = std::min((int)(ox * scaleX), w - 1);
            int iy = std::min((int)(oy * scaleY), h - 1);
            for (int ch = 0; ch < 3; ++ch) {
                float raw = px[(iy * w + ix) * 3 + ch] / 255.0f;
                img[(oy * OUT + ox) * 3 + ch] = (raw - 0.5f) / 0.5f;  // → [-1, 1]
            }
        }
    }

    stbi_image_free(px);
    return img;  // NHWC [64, 64, 3]
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

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    const std::string imagePath  = "img/9961.jpg";
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

    Relu relu1(ctx, 64 * 64 * 16);
    auto out_relu1 = relu1.run(out_conv1);      // [64*64*16]

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