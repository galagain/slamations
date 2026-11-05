#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "SemanticMasker.h"

using ORB_SLAM3::SemanticMasker;
using ORB_SLAM3::SemanticMaskerConfig;

namespace
{

struct ImageEntry
{
    double timestamp = 0.0;
    std::string rgbPath;
};

double NowMs()
{
    typedef std::chrono::steady_clock Clock;
    static const Clock::time_point start = Clock::now();
    const Clock::time_point now = Clock::now();
    return std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(now - start).count();
}

bool ReadBool(cv::FileStorage &fs, const std::string &key, bool fallback)
{
    cv::FileNode node = fs[key];
    if(node.empty())
        return fallback;
    int value = fallback ? 1 : 0;
    node >> value;
    return value != 0;
}

int ReadInt(cv::FileStorage &fs, const std::string &key, int fallback)
{
    cv::FileNode node = fs[key];
    if(node.empty())
        return fallback;
    int value = fallback;
    node >> value;
    return value;
}

float ReadFloat(cv::FileStorage &fs, const std::string &key, float fallback)
{
    cv::FileNode node = fs[key];
    if(node.empty())
        return fallback;
    float value = fallback;
    node >> value;
    return value;
}

std::string ReadString(cv::FileStorage &fs, const std::string &key, const std::string &fallback)
{
    cv::FileNode node = fs[key];
    if(node.empty())
        return fallback;
    std::string value;
    node >> value;
    return value.empty() ? fallback : value;
}

SemanticMaskerConfig LoadSemanticConfig(const std::string &settingsPath)
{
    cv::FileStorage fs(settingsPath, cv::FileStorage::READ);
    if(!fs.isOpened())
        throw std::runtime_error("could not open settings: " + settingsPath);

    SemanticMaskerConfig config;
    config.enabled = true;
    config.backend = ReadString(fs, "Semantic.backend", config.backend);
    config.segmentationRoot = ReadString(fs, "Semantic.segmentation_root", config.segmentationRoot);
    config.segnetEnginePath = ReadString(fs, "Semantic.segnet_engine", config.segnetEnginePath);
    config.segnetPluginPath = ReadString(fs, "Semantic.segnet_plugin", config.segnetPluginPath);
    config.maskrcnnEnginePath = ReadString(fs, "Semantic.maskrcnn_engine", config.maskrcnnEnginePath);
    config.tokenmaskModelPath = ReadString(fs, "Semantic.tokenmask_model", config.tokenmaskModelPath);
    config.tokenmaskEnginePath = ReadString(fs, "Semantic.tokenmask_engine", config.tokenmaskEnginePath);
    config.tokenmaskDataset = ReadString(fs, "Semantic.tokenmask_dataset", config.tokenmaskDataset);
    config.dynamicLabels = ReadString(fs, "Semantic.dynamic_labels", config.dynamicLabels);
    config.deviceId = ReadInt(fs, "Semantic.device_id", config.deviceId);
    config.dilatePixels = ReadInt(fs, "Semantic.dilate_pixels", config.dilatePixels);
    config.inputIsRgb = ReadBool(fs, "Semantic.input_is_rgb", config.inputIsRgb);
    config.segnetThreshold = ReadFloat(fs, "Semantic.segnet_threshold", config.segnetThreshold);
    config.maskrcnnThreshold = ReadFloat(fs, "Semantic.maskrcnn_threshold", config.maskrcnnThreshold);
    config.maskrcnnMinSide = ReadInt(fs, "Semantic.maskrcnn_min_side", config.maskrcnnMinSide);
    config.tokenmaskInputWidth = ReadInt(fs, "Semantic.tokenmask_input_width", config.tokenmaskInputWidth);
    config.tokenmaskInputHeight = ReadInt(fs, "Semantic.tokenmask_input_height", config.tokenmaskInputHeight);
    config.tokenmaskMaskThreshold =
        ReadFloat(fs, "Semantic.tokenmask_mask_threshold", config.tokenmaskMaskThreshold);
    config.tokenmaskOverlapThreshold =
        ReadFloat(fs, "Semantic.tokenmask_overlap_threshold", config.tokenmaskOverlapThreshold);

    // This benchmark measures pure per-image semantic prediction. It therefore
    // disables stride/reuse even when the SLAM preset uses sparse inference.
    config.frameStride = 1;
    config.maxMaskAge = 0;
    config.submitBeforeOrb = false;
    config.skipUntilInitialized = false;
    return config;
}

std::vector<ImageEntry> LoadAssociations(const std::string &sequenceDir, const std::string &assocPath)
{
    std::ifstream file(assocPath.c_str());
    if(!file)
        throw std::runtime_error("could not open associations: " + assocPath);

    std::vector<ImageEntry> entries;
    std::string line;
    while(std::getline(file, line))
    {
        if(line.empty() || line[0] == '#')
            continue;
        std::istringstream iss(line);
        ImageEntry entry;
        std::string depthStamp;
        std::string depthPath;
        if(!(iss >> entry.timestamp >> entry.rgbPath >> depthStamp >> depthPath))
            continue;
        entry.rgbPath = sequenceDir + "/" + entry.rgbPath;
        entries.push_back(entry);
    }
    return entries;
}

double Percentile(std::vector<double> values, double q)
{
    if(values.empty())
        return 0.0;
    std::sort(values.begin(), values.end());
    const double pos = (values.size() - 1) * q;
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = static_cast<size_t>(std::ceil(pos));
    if(lo == hi)
        return values[lo];
    const double alpha = pos - static_cast<double>(lo);
    return values[lo] * (1.0 - alpha) + values[hi] * alpha;
}

double Mean(const std::vector<double> &values)
{
    if(values.empty())
        return 0.0;
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

} // namespace

int main(int argc, char **argv)
{
    if(argc < 5)
    {
        std::cerr << "usage: semantic_tum_benchmark settings.yaml sequence_dir associations.txt output_dir [limit]\n";
        return 2;
    }

    const std::string settingsPath = argv[1];
    const std::string sequenceDir = argv[2];
    const std::string assocPath = argv[3];
    const std::string outputDir = argv[4];
    const int limit = argc >= 6 ? std::max(0, std::atoi(argv[5])) : 0;

    try
    {
        SemanticMaskerConfig config = LoadSemanticConfig(settingsPath);
        SemanticMasker masker(config);
        if(!masker.Start())
            throw std::runtime_error("SemanticMasker did not start");

        std::vector<ImageEntry> entries = LoadAssociations(sequenceDir, assocPath);
        if(limit > 0 && static_cast<size_t>(limit) < entries.size())
            entries.resize(static_cast<size_t>(limit));
        if(entries.empty())
            throw std::runtime_error("no images to benchmark");

        std::vector<double> readTimes;
        std::vector<double> semanticTimes;
        readTimes.reserve(entries.size());
        semanticTimes.reserve(entries.size());

        const std::string csvPath = outputDir + "/semantic_only_times.csv";
        std::ofstream csv(csvPath.c_str());
        csv << "frame,timestamp,read_ms,semantic_ms,total_ms,mask_nonzero\n";

        const double totalStart = NowMs();
        for(size_t i = 0; i < entries.size(); ++i)
        {
            const double readStart = NowMs();
            cv::Mat image = cv::imread(entries[i].rgbPath, cv::IMREAD_COLOR);
            const double readEnd = NowMs();
            if(image.empty())
                throw std::runtime_error("could not read image: " + entries[i].rgbPath);

            const double semanticStart = NowMs();
            const unsigned long long requestId = masker.Submit(image, entries[i].timestamp);
            cv::Mat mask;
            if(!masker.WaitForMask(requestId, mask))
                throw std::runtime_error("semantic prediction failed");
            const double semanticEnd = NowMs();

            const double readMs = readEnd - readStart;
            const double semanticMs = semanticEnd - semanticStart;
            const double totalMs = semanticEnd - readStart;
            readTimes.push_back(readMs);
            semanticTimes.push_back(semanticMs);
            csv << i << "," << std::fixed << std::setprecision(6) << entries[i].timestamp << ","
                << readMs << "," << semanticMs << "," << totalMs << ","
                << cv::countNonZero(mask) << "\n";
        }
        const double totalEnd = NowMs();
        masker.Stop();

        const double semanticMean = Mean(semanticTimes);
        const double readMean = Mean(readTimes);
        const double totalWallMs = totalEnd - totalStart;
        const double fps = semanticMean > 0.0 ? 1000.0 / semanticMean : 0.0;
        const double wallFps = totalWallMs > 0.0 ? entries.size() * 1000.0 / totalWallMs : 0.0;

        const std::string summaryPath = outputDir + "/semantic_only_summary.txt";
        std::ofstream summary(summaryPath.c_str());
        summary << "backend " << config.backend << "\n";
        summary << "frames " << entries.size() << "\n";
        summary << "read_mean_ms " << readMean << "\n";
        summary << "semantic_mean_ms " << semanticMean << "\n";
        summary << "semantic_p50_ms " << Percentile(semanticTimes, 0.50) << "\n";
        summary << "semantic_p95_ms " << Percentile(semanticTimes, 0.95) << "\n";
        summary << "semantic_p99_ms " << Percentile(semanticTimes, 0.99) << "\n";
        summary << "semantic_fps " << fps << "\n";
        summary << "wall_total_s " << totalWallMs / 1000.0 << "\n";
        summary << "wall_fps_including_imread " << wallFps << "\n";

        std::cout << summary.rdbuf();
        return 0;
    }
    catch(const std::exception &exc)
    {
        std::cerr << "semantic_tum_benchmark failed: " << exc.what() << std::endl;
        return 1;
    }
}
