#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <string>
#include <iostream>

// ==========================================================================
// Minimal JSON parser – reads flat key:integer pairs from meta.json
// ==========================================================================
static int jsonGetInt(const std::string& json, const std::string& key)
{
    std::string pattern = "\"" + key + "\"";
    auto pos = json.find(pattern);
    if (pos == std::string::npos)
        throw std::runtime_error("meta.json: key '" + key + "' not found");
    pos = json.find(':', pos);
    if (pos == std::string::npos)
        throw std::runtime_error("meta.json: malformed entry for '" + key + "'");
    ++pos;
    while (pos < json.size() &&
           (json[pos] == ' ' || json[pos] == '\n' || json[pos] == '\r'))
        ++pos;
    return std::stoi(json.substr(pos));
}

// ==========================================================================
// KernelMeta – all dimensions/strides read from meta.json
// ==========================================================================
struct KernelMeta {
    int kH, kW;
    int inC, outC;
    int padH,    padW;
    int strideH, strideW;
};

// ==========================================================================
// loadKernelFromFiles
//   Reads weights/meta.json and weights/kernel.bin produced by
//   extract_weights.py, returns the flat float32 weight vector and fills
//   a KernelMeta struct with every dimension the convolution needs.
// ==========================================================================
static std::vector<float> loadKernelFromFiles(
    const std::string& metaPath,
    const std::string& kernelPath,
    KernelMeta&        meta)
{
    // ---- 1. Parse meta.json ----
    std::ifstream mf(metaPath);
    if (!mf.is_open())
        throw std::runtime_error("Cannot open " + metaPath);
    std::ostringstream ss;
    ss << mf.rdbuf();
    const std::string json = ss.str();

    meta.kH      = jsonGetInt(json, "kH");
    meta.kW      = jsonGetInt(json, "kW");
    meta.inC     = jsonGetInt(json, "inC");
    meta.outC    = jsonGetInt(json, "outC");
    meta.padH    = jsonGetInt(json, "padH");
    meta.padW    = jsonGetInt(json, "padW");
    meta.strideH = jsonGetInt(json, "strideH");
    meta.strideW = jsonGetInt(json, "strideW");

    std::cout << "Kernel meta:"
              << "  kH="      << meta.kH      << " kW="      << meta.kW
              << "  inC="     << meta.inC      << " outC="    << meta.outC
              << "  padH="    << meta.padH     << " padW="    << meta.padW
              << "  strideH=" << meta.strideH  << " strideW=" << meta.strideW
              << "\n";

    // ---- 2. Read kernel.bin ----
    std::ifstream kf(kernelPath, std::ios::binary | std::ios::ate);
    if (!kf.is_open())
        throw std::runtime_error("Cannot open " + kernelPath);

    const std::streamsize byteCount = kf.tellg();
    kf.seekg(0, std::ios::beg);

    const size_t expected =
        static_cast<size_t>(meta.kH) * meta.kW * meta.inC * meta.outC;

    if (byteCount != static_cast<std::streamsize>(expected * sizeof(float)))
        throw std::runtime_error(
            "kernel.bin size mismatch: expected " +
            std::to_string(expected * sizeof(float)) +
            " bytes, got " + std::to_string(byteCount));

    std::vector<float> weights(expected);
    kf.read(reinterpret_cast<char*>(weights.data()),
            static_cast<std::streamsize>(expected * sizeof(float)));

    if (!kf)
        throw std::runtime_error("Failed to read kernel.bin completely");

    std::cout << "Loaded " << expected
              << " kernel weights from " << kernelPath << "\n";
    return weights;
}
