/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef ORBEXTRACTOR_H
#define ORBEXTRACTOR_H

#include <vector>
#include <list>
#include <string>
#include <cstdlib>
#include <opencv2/opencv.hpp>

#ifdef ORB_SLAM3_USE_CUDA_FRONTEND
#include "cuda/orb_gpu.hpp"
#endif

namespace ORB_SLAM3
{

class ExtractorNode
{
public:
    ExtractorNode():bNoMore(false){}

    void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

    std::vector<cv::KeyPoint> vKeys;
    cv::Point2i UL, UR, BL, BR;
    std::list<ExtractorNode>::iterator lit;
    bool bNoMore;
};

struct ORBExtractorConfig
{
    enum DistributionMode
    {
        DISTRIBUTION_OCTREE = 0,
        DISTRIBUTION_GRID = 1
    };

    enum FastDetectorType
    {
        FAST_9_16 = 0,
        FAST_7_12 = 1,
        FAST_5_8 = 2
    };

    ORBExtractorConfig()
        : distributionMode(DISTRIBUTION_OCTREE),
          fastDetectorType(FAST_9_16),
          tileWidth(35),
          tileHeight(35),
          fixedMultiScaleTileSize(true),
          maxKeypointsPerTile(0),
          maxPyramidLevels(0),
          adaptiveFeatureRescue(false),
          adaptiveMinKeypoints(0),
          adaptiveMinKeypointRatio(0.0f),
          adaptiveIniThFAST(0),
          adaptiveMinThFAST(0),
          adaptiveUseFullPyramid(false),
          adaptiveAllowCpuFallbackWithCuda(false),
          useCudaFrontend(false),
          useCudaStereoMatching(false),
          useCudaFrustumProjection(false),
          cudaIsJetson(false),
          cudaDeviceId(0),
          fastNMin(0),
          fastNMax(0),
          applyNmsMs(false),
          nmsMsModeGpu(false)
    {
    }

    DistributionMode distributionMode;
    FastDetectorType fastDetectorType;
    int tileWidth;
    int tileHeight;
    bool fixedMultiScaleTileSize;
    int maxKeypointsPerTile;
    int maxPyramidLevels;
    bool adaptiveFeatureRescue;
    int adaptiveMinKeypoints;
    float adaptiveMinKeypointRatio;
    int adaptiveIniThFAST;
    int adaptiveMinThFAST;
    bool adaptiveUseFullPyramid;
    bool adaptiveAllowCpuFallbackWithCuda;
    bool useCudaFrontend;
    bool useCudaStereoMatching;
    bool useCudaFrustumProjection;
    bool cudaIsJetson;
    int cudaDeviceId;
    int fastNMin;
    int fastNMax;
    bool applyNmsMs;
    bool nmsMsModeGpu;
};

inline bool AntSlamEnvBool(const char *name, bool fallback)
{
    const char *value = std::getenv(name);
    if(!value || !*value)
        return fallback;

    const std::string text(value);
    if(text == "0" || text == "false" || text == "False" || text == "FALSE" ||
       text == "no" || text == "No" || text == "NO" || text == "off" ||
       text == "Off" || text == "OFF")
        return false;

    return true;
}

inline int AntSlamEnvInt(const char *name, int fallback)
{
    const char *value = std::getenv(name);
    if(!value || !*value)
        return fallback;

    char *end = 0;
    const long parsed = std::strtol(value, &end, 10);
    if(end == value)
        return fallback;
    return static_cast<int>(parsed);
}

inline float AntSlamEnvFloat(const char *name, float fallback)
{
    const char *value = std::getenv(name);
    if(!value || !*value)
        return fallback;

    char *end = 0;
    const float parsed = std::strtof(value, &end);
    if(end == value)
        return fallback;
    return parsed;
}

inline void ForceLiteSemSLAMOrbOnlyConfig(ORBExtractorConfig &config)
{
    const int cudaDeviceId = config.cudaDeviceId;
    config.useCudaFrontend = true;
    config.useCudaStereoMatching = false;
    config.useCudaFrustumProjection = false;
    config.adaptiveAllowCpuFallbackWithCuda = false;
    config.cudaIsJetson = true;
    config.cudaDeviceId = cudaDeviceId;

    const int tileSize = AntSlamEnvInt("ANTSLAM_CUDA_ORB_TILE", 0);
    if(tileSize > 0)
    {
        config.tileWidth = tileSize;
        config.tileHeight = tileSize;
    }

    config.useCudaFrontend = AntSlamEnvBool("ANTSLAM_CUDA_FRONTEND", config.useCudaFrontend);
    config.tileWidth = AntSlamEnvInt("ANTSLAM_CUDA_ORB_TILE_WIDTH", config.tileWidth);
    config.tileHeight = AntSlamEnvInt("ANTSLAM_CUDA_ORB_TILE_HEIGHT", config.tileHeight);
    config.fixedMultiScaleTileSize = AntSlamEnvBool("ANTSLAM_CUDA_ORB_FIXED_TILE", config.fixedMultiScaleTileSize);
    config.fastNMin = AntSlamEnvInt("ANTSLAM_CUDA_FAST_N_MIN", config.fastNMin);
    config.fastNMax = AntSlamEnvInt("ANTSLAM_CUDA_FAST_N_MAX", config.fastNMax);
    config.applyNmsMs = AntSlamEnvBool("ANTSLAM_CUDA_APPLY_NMS_MS", config.applyNmsMs);
    config.nmsMsModeGpu = AntSlamEnvBool("ANTSLAM_CUDA_NMS_MS_GPU", config.nmsMsModeGpu);
    config.adaptiveFeatureRescue = AntSlamEnvBool("ANTSLAM_CUDA_ADAPTIVE_RESCUE", config.adaptiveFeatureRescue);
    config.adaptiveMinKeypoints = AntSlamEnvInt("ANTSLAM_CUDA_ADAPTIVE_MIN_KEYPOINTS", config.adaptiveMinKeypoints);
    config.adaptiveMinKeypointRatio = AntSlamEnvFloat("ANTSLAM_CUDA_ADAPTIVE_MIN_RATIO", config.adaptiveMinKeypointRatio);
    config.adaptiveIniThFAST = AntSlamEnvInt("ANTSLAM_CUDA_ADAPTIVE_INI_FAST", config.adaptiveIniThFAST);
    config.adaptiveMinThFAST = AntSlamEnvInt("ANTSLAM_CUDA_ADAPTIVE_MIN_FAST", config.adaptiveMinThFAST);
    config.adaptiveUseFullPyramid = AntSlamEnvBool("ANTSLAM_CUDA_ADAPTIVE_FULL_PYRAMID", config.adaptiveUseFullPyramid);
    config.adaptiveAllowCpuFallbackWithCuda = AntSlamEnvBool("ANTSLAM_CUDA_ADAPTIVE_CPU_FALLBACK", config.adaptiveAllowCpuFallbackWithCuda);
    config.cudaIsJetson = AntSlamEnvBool("ANTSLAM_CUDA_IS_JETSON", config.cudaIsJetson);
    config.cudaDeviceId = AntSlamEnvInt("ANTSLAM_CUDA_DEVICE_ID", config.cudaDeviceId);
}

class ORBextractor
{
public:
    
    enum {HARRIS_SCORE=0, FAST_SCORE=1 };

    ORBextractor(int nfeatures, float scaleFactor, int nlevels,
                 int iniThFAST, int minThFAST,
                 const ORBExtractorConfig &config = ORBExtractorConfig(),
                 const std::string &maskPath = std::string());

    ~ORBextractor();

    // Compute the ORB features and descriptors on an image.
    // ORB are dispersed on the image using an octree.
    // Mask is ignored in the current implementation.
    int operator()( cv::InputArray _image, cv::InputArray _mask,
                    std::vector<cv::KeyPoint>& _keypoints,
                    cv::OutputArray _descriptors, std::vector<int> &vLappingArea);
    int ExtractCPUOnly(cv::InputArray _image, cv::InputArray _mask,
                       std::vector<cv::KeyPoint>& _keypoints,
                       cv::OutputArray _descriptors, std::vector<int> &vLappingArea);

    bool UsesCUDAFrontend() const { return mbUseCudaFrontend; }
    const std::string& GetMaskPath() const { return mMaskPath; }
    void SetFeatureBudget(int targetFeatures);
    int GetFeatureBudget() const { return nfeatures; }

#ifdef ORB_SLAM3_USE_CUDA_FRONTEND
    void ExtractCuda(const cv::Mat &image,
                     orb_cuda::SyncedMem<int> &keypoints,
                     orb_cuda::SyncedMem<unsigned char> &descriptors,
                     const cv::Mat &mask = cv::Mat());
    orb_cuda::ORB_GPU* GetCudaORB() { return mpCudaORBExtractor; }
    const orb_cuda::ORB_GPU* GetCudaORB() const { return mpCudaORBExtractor; }
#endif

    int inline GetLevels(){
        return nlevels;}

    float inline GetScaleFactor(){
        return scaleFactor;}

    std::vector<float> inline GetScaleFactors(){
        return mvScaleFactor;
    }

    std::vector<float> inline GetInverseScaleFactors(){
        return mvInvScaleFactor;
    }

    std::vector<float> inline GetScaleSigmaSquares(){
        return mvLevelSigma2;
    }

    std::vector<float> inline GetInverseScaleSigmaSquares(){
        return mvInvLevelSigma2;
    }

    std::vector<cv::Mat> mvImagePyramid;

protected:

    int GetTileWidth(int level) const;
    int GetTileHeight(int level) const;
    int GetActiveLevels() const;
    int GetAdaptiveRescueMinKeypoints() const;
    bool ShouldRunAdaptiveFeatureRescue(int extractedKeypoints) const;
    cv::FastFeatureDetector::DetectorType GetFastDetectorTypeCV() const;
    void DetectFastKeypoints(const cv::Mat &cellImage, std::vector<cv::KeyPoint> &keypoints) const;
    void ComputePyramid(cv::Mat image);
    void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);    
    std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
                                           const int &maxX, const int &minY, const int &maxY, const int &nFeatures, const int &level);

    void ComputeKeyPointsOld(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);
    int ExtractWithCPU(const cv::Mat &image,
                       std::vector<cv::KeyPoint>& keypoints,
                       cv::OutputArray descriptors,
                       std::vector<int> &vLappingArea);
    std::vector<cv::Point> pattern;

    int nfeatures;
    double scaleFactor;
    int nlevels;
    int iniThFAST;
    int minThFAST;
    ORBExtractorConfig mConfig;
    bool mbUseCudaFrontend;
    std::string mMaskPath;

    std::vector<int> mnFeaturesPerLevel;

    std::vector<int> umax;

    std::vector<float> mvScaleFactor;
    std::vector<float> mvInvScaleFactor;    
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;

#ifdef ORB_SLAM3_USE_CUDA_FRONTEND
    void EnsureCudaExtractorForImage(const cv::Mat &image);
    void ConvertCudaOutputToCv(orb_cuda::SyncedMem<int> &keypoints,
                               orb_cuda::SyncedMem<unsigned char> &descriptors,
                               std::vector<cv::KeyPoint> &outputKeypoints,
                               cv::Mat &outputDescriptors) const;
    orb_cuda::ORB_GPU* mpCudaORBExtractor;
    int mCudaImageHeight;
    int mCudaImageWidth;
#endif
};

} //namespace ORB_SLAM

#endif
