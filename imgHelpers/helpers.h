// ---- stb (implement exactly once per translation unit) -------------------
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
 
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"
// --------------------------------------------------------------------------

#include <algorithm>
#include <stdexcept>

// ==========================================================================
// Image helpers
// ==========================================================================
 
/**
 * Load an image from disk into a flat float vector (HWC, values in [0,1]).
 *
 * @param path     File path (PNG / JPG / BMP / TGA / …)
 * @param outH     Populated with image height in pixels
 * @param outW     Populated with image width  in pixels
 * @param outC     Populated with number of channels  (1, 3, or 4)
 * @param forceC   If > 0, force stb to decode to this many channels
 *                 (e.g. forceC=3 → always RGB even for RGBA source)
 * @return         Flat vector in row-major HWC order, dtype float, [0,1]
 */
std::vector<float> loadImage(const std::string& path,
                              int& outH, int& outW, int& outC,
                              int forceC = 0)
{
    int w, h, c;
    uint8_t* data = stbi_load(path.c_str(), &w, &h, &c, forceC);
    if (!data)
        throw std::runtime_error("stbi_load failed: " + std::string(stbi_failure_reason()));
 
    outW = w;
    outH = h;
    outC = (forceC > 0) ? forceC : c;
 
    const size_t n = static_cast<size_t>(outH) * outW * outC;
    std::vector<float> buf(n);
    for (size_t i = 0; i < n; ++i)
        buf[i] = data[i] / 255.0f;          // normalise to [0,1]
 
    stbi_image_free(data);
    return buf;
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
