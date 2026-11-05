#ifndef ORB_SLAM3_SEMANTIC_MASKER_H
#define ORB_SLAM3_SEMANTIC_MASKER_H

#include <memory>
#include <set>
#include <string>

#include <opencv2/core/core.hpp>

namespace ORB_SLAM3
{

struct SemanticMaskerConfig
{
    bool enabled = false;
    std::string backend = "segnet";
    std::string segmentationRoot = "/home/nvidia/Desktop/kalvici/segmentation";
    std::string segnetEnginePath;
    std::string segnetPluginPath;
    std::string maskrcnnEnginePath;
    std::string tokenmaskModelPath;
    std::string tokenmaskEnginePath;
    std::string tokenmaskDataset = "ade20k";
    std::string dynamicLabels;
    int deviceId = 0;
    int dilatePixels = 3;
    int frameStride = 1;
    int maxMaskAge = 0;
    bool submitBeforeOrb = false;
    bool skipUntilInitialized = false;
    int minKeptFeatures = 0;
    float minKeepRatio = 0.0f;
    bool inputIsRgb = false;
    float segnetThreshold = 0.8f;
    float maskrcnnThreshold = 0.7f;
    int maskrcnnMinSide = 512;
    int tokenmaskInputWidth = 640;
    int tokenmaskInputHeight = 640;
    float tokenmaskMaskThreshold = 0.8f;
    float tokenmaskOverlapThreshold = 0.8f;
};

class SemanticMasker
{
public:
    explicit SemanticMasker(const SemanticMaskerConfig &config);
    ~SemanticMasker();

    bool Start();
    void Stop();
    bool IsEnabled() const;

    unsigned long long Submit(const cv::Mat &image, double timestamp);
    bool WaitForMask(unsigned long long requestId, cv::Mat &featureMask);

private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace ORB_SLAM3

#endif
