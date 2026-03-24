#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <string>
#include <iostream>

// ==========================================================================
// Minimal JSON parser
// ==========================================================================
static int jsonGetInt(const std::string& json, const std::string& key)
{
    std::string pattern = "\"" + key + "\"";
    auto pos = json.find(pattern);
    if (pos == std::string::npos)
        throw std::runtime_error("meta.json: key '" + key + "' not found");
    pos = json.find(':', pos);
    ++pos;
    while (pos < json.size() &&
           (json[pos] == ' ' || json[pos] == '\n' || json[pos] == '\r'))
        ++pos;
    return std::stoi(json.substr(pos));
}

// ==========================================================================
// LinearMeta
// ==========================================================================
struct LinearMeta {
    int  in_features;
    int  out_features;
    bool has_bias;
};

// ==========================================================================
// Load a raw float32 binary file into a vector
// ==========================================================================
static std::vector<float> loadBin(const std::string& path, size_t expectedFloats)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open())
        throw std::runtime_error("Cannot open " + path);

    const std::streamsize bytes = f.tellg();
    f.seekg(0, std::ios::beg);

    if (bytes != static_cast<std::streamsize>(expectedFloats * sizeof(float)))
        throw std::runtime_error(
            path + " size mismatch: expected " +
            std::to_string(expectedFloats * sizeof(float)) +
            " bytes, got " + std::to_string(bytes));

    std::vector<float> data(expectedFloats);
    f.read(reinterpret_cast<char*>(data.data()),
           static_cast<std::streamsize>(expectedFloats * sizeof(float)));
    if (!f)
        throw std::runtime_error("Failed to read " + path);

    std::cout << "Loaded " << expectedFloats << " floats from " << path << "\n";
    return data;
}

// ==========================================================================
// loadLinearWeights
// ==========================================================================
static void loadLinearWeights(
    const std::string&   weightsDir,
    LinearMeta&          meta,
    std::vector<float>&  weight,   // [out_features, in_features]
    std::vector<float>&  bias)     // [out_features]  (empty if no bias)
{
    // ---- meta.json ----
    const std::string metaPath = weightsDir + "/meta.json";
    std::ifstream mf(metaPath);
    if (!mf.is_open())
        throw std::runtime_error("Cannot open " + metaPath);
    std::ostringstream ss;
    ss << mf.rdbuf();
    const std::string json = ss.str();

    meta.in_features  = jsonGetInt(json, "in_features");
    meta.out_features = jsonGetInt(json, "out_features");
    meta.has_bias     = jsonGetInt(json, "has_bias") != 0;

    std::cout << "Linear meta:"
              << "  in_features="  << meta.in_features
              << "  out_features=" << meta.out_features
              << "  has_bias="     << meta.has_bias << "\n";

    // ---- weight.bin  [out_features * in_features] ----
    weight = loadBin(weightsDir + "/weight.bin",
                     static_cast<size_t>(meta.out_features) * meta.in_features);

    // ---- bias.bin  [out_features]  (optional) ----
    if (meta.has_bias)
        bias = loadBin(weightsDir + "/bias.bin",
                       static_cast<size_t>(meta.out_features));
    else
        bias.clear();
}