// ---- stb (implement exactly once per translation unit) -------------------
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
 
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb/stb_image_resize.h"
// --------------------------------------------------------------------------

#include <algorithm>
#include <stdexcept>

// ── Image loader & preprocessor ───────────────────────────────────────────────
// Simple bilinear resize (matches PyTorch INTER_LINEAR / antialias=False)
std::vector<float> bilinearResize(const std::vector<float>& src, 
                                   int src_w, int src_h, int channels,
                                   int dst_w, int dst_h) {
    std::vector<float> dst(dst_w * dst_h * channels);

    float x_scale = (float)src_w / dst_w;
    float y_scale = (float)src_h / dst_h;

    for (int c = 0; c < channels; ++c) {
        for (int dy = 0; dy < dst_h; ++dy) {
            for (int dx = 0; dx < dst_w; ++dx) {
                // Map destination pixel to source space
                float sx = (dx + 0.5f) * x_scale - 0.5f;
                float sy = (dy + 0.5f) * y_scale - 0.5f;

                int x0 = (int)std::floor(sx), x1 = std::min(x0 + 1, src_w - 1);
                int y0 = (int)std::floor(sy), y1 = std::min(y0 + 1, src_h - 1);
                x0 = std::max(x0, 0);
                y0 = std::max(y0, 0);

                float wx = sx - std::floor(sx);
                float wy = sy - std::floor(sy);

                // Bilinear interpolation
                float v00 = src[(y0 * src_w + x0) * channels + c];
                float v10 = src[(y0 * src_w + x1) * channels + c];
                float v01 = src[(y1 * src_w + x0) * channels + c];
                float v11 = src[(y1 * src_w + x1) * channels + c];

                dst[(dy * dst_w + dx) * channels + c] =
                    (1 - wy) * ((1 - wx) * v00 + wx * v10) +
                    wy  * ((1 - wx) * v01 + wx * v11);
            }
        }
    }
    return dst;
}

// Returns flat float buffer in CHW order, shape [3, 64, 64]
std::vector<float> loadImage(const std::string& image_path,
                                    int target_w = 64, int target_h = 64) {
    // --- Load image as RGB uint8 ---
    int src_w, src_h, channels;
    unsigned char* raw = stbi_load(image_path.c_str(), &src_w, &src_h, &channels, 3);
    if (!raw) {
        throw std::runtime_error("Failed to load image: " + image_path);
    }
    channels = 3; // forced RGB via the last arg above

    // Print max/min like Python does
    auto [mn, mx] = std::minmax_element(raw, raw + src_w * src_h * channels);
    std::cout << "STB Max min: " << (int)*mx << " " << (int)*mn << std::endl;

    // --- ToTensor(): uint8 [0,255] -> float32 [0.0, 1.0], HWC layout ---
    std::vector<float> float_img(src_w * src_h * channels);
    for (int i = 0; i < src_w * src_h * channels; ++i) {
        float_img[i] = raw[i] / 255.0f;
    }
    stbi_image_free(raw);

    // --- Resize (64, 64) with bilinear, antialias=False ---
    std::vector<float> resized = bilinearResize(float_img, src_w, src_h, channels,
                                                 target_w, target_h);

    // --- Normalize((0.5,), (0.5,)): out = (in - 0.5) / 0.5 ---
    // --- Convert HWC -> CHW ---
    std::vector<float> tensor(channels * target_h * target_w);
    for (int c = 0; c < channels; ++c) {
        for (int y = 0; y < target_h; ++y) {
            for (int x = 0; x < target_w; ++x) {
                float val = resized[(y * target_w + x) * channels + c];
                val = (val - 0.5f) / 0.5f;  // normalize
                tensor[c * target_h * target_w + y * target_w + x] = val;
            }
        }
    }

    float minVal = *std::min_element(tensor.begin(), tensor.end());
    float maxVal = *std::max_element(tensor.begin(), tensor.end());
    std::cout << "Resized - Normalized Max min: " << maxVal << " " << minVal << std::endl;

    return tensor;  // CHW float32, values in [-1, 1]
}

/**
 * Save a float HWC buffer as a PNG.
 *
 * Values are clamped to [0,1] then scaled to [0,255].
 *
 * @param path  Output file path (must end in .png for PNG output)
 * @param buf   Flat HWC float buffer
 * @param H     Height in pixels
 * @param W     Width  in pixels
 * @param C     Channels  (1 = grey, 3 = RGB, 4 = RGBA)
 */
void saveImagePNG(const std::string& path,
                  const std::vector<float>& buf,
                  int H, int W, int C)
{
    const size_t n = static_cast<size_t>(H) * W * C;
    if (buf.size() < n)
        throw std::runtime_error("saveImagePNG: buffer too small");
 
    std::vector<uint8_t> pixels(n);
    for (size_t i = 0; i < n; ++i) {
        float v = std::clamp(buf[i], 0.0f, 1.0f);
        pixels[i] = static_cast<uint8_t>(v * 255.0f + 0.5f);
    }
 
    // stride_in_bytes = W * C  (tightly packed)
    if (!stbi_write_png(path.c_str(), W, H, C, pixels.data(), W * C))
        throw std::runtime_error("stbi_write_png failed for: " + path);
}
