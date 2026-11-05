#include "Kernels/AntSlamCudaLocalMatcher.h"

#ifdef ORB_SLAM3_USE_CUDA_FRONTEND

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

#include <cuda_runtime_api.h>

namespace
{

constexpr int kGridCols = 64;
constexpr int kGridRows = 48;
constexpr int kKeypointsPerCell = 20;
constexpr int kDescriptorSize = 32;

extern "C" void AntSlamLaunchLocalSearchKernel(
    const unsigned char *d_active,
    const int *d_scale,
    const float *d_viewCos,
    const float *d_projX,
    const float *d_projY,
    const float *d_projXR,
    const unsigned char *d_mpDesc,
    const float *d_keyX,
    const float *d_keyY,
    const int *d_keyOctave,
    const unsigned char *d_frameDesc,
    const float *d_mvuRight,
    const unsigned char *d_frameOccupied,
    const int *d_grid,
    const int *d_gridSize,
    const float *d_scaleFactors,
    int numPoints,
    int numFeatures,
    int numLevels,
    float minX,
    float minY,
    float gridElementWidthInv,
    float gridElementHeightInv,
    float th,
    int *d_bestLevel,
    int *d_bestLevel2,
    int *d_bestDist,
    int *d_bestDist2,
    int *d_bestIdx);

bool EnvBool(const char *name, bool fallback)
{
    const char *value = std::getenv(name);
    if(!value || !*value)
        return fallback;
    const std::string text(value);
    return !(text == "0" || text == "false" || text == "False" ||
             text == "FALSE" || text == "off" || text == "OFF" ||
             text == "no" || text == "NO");
}

void CheckCuda(cudaError_t err, const char *what)
{
    if(err != cudaSuccess)
    {
        std::cerr << "AntSLAM CUDA local matcher: " << what << ": "
                  << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(what);
    }
}

template<typename T>
void FreePtr(T *&ptr)
{
    if(ptr)
    {
        cudaFree(ptr);
        ptr = nullptr;
    }
}

template<typename T>
void AllocatePtr(T *&ptr, size_t count, const char *what)
{
    CheckCuda(cudaMalloc(reinterpret_cast<void **>(&ptr), count * sizeof(T)), what);
}

struct DeviceBuffers
{
    int pointCapacity = 0;
    int featureCapacity = 0;
    int levelCapacity = 0;

    unsigned char *d_active = nullptr;
    int *d_scale = nullptr;
    float *d_viewCos = nullptr;
    float *d_projX = nullptr;
    float *d_projY = nullptr;
    float *d_projXR = nullptr;
    unsigned char *d_mpDesc = nullptr;

    float *d_keyX = nullptr;
    float *d_keyY = nullptr;
    int *d_keyOctave = nullptr;
    unsigned char *d_frameDesc = nullptr;
    float *d_mvuRight = nullptr;
    unsigned char *d_frameOccupied = nullptr;

    int *d_grid = nullptr;
    int *d_gridSize = nullptr;
    float *d_scaleFactors = nullptr;

    int *d_bestLevel = nullptr;
    int *d_bestLevel2 = nullptr;
    int *d_bestDist = nullptr;
    int *d_bestDist2 = nullptr;
    int *d_bestIdx = nullptr;

    void Ensure(int numPoints, int numFeatures, int numLevels)
    {
        if(numPoints > pointCapacity)
        {
            FreePtr(d_active);
            FreePtr(d_scale);
            FreePtr(d_viewCos);
            FreePtr(d_projX);
            FreePtr(d_projY);
            FreePtr(d_projXR);
            FreePtr(d_mpDesc);
            FreePtr(d_bestLevel);
            FreePtr(d_bestLevel2);
            FreePtr(d_bestDist);
            FreePtr(d_bestDist2);
            FreePtr(d_bestIdx);

            pointCapacity = std::max(numPoints, pointCapacity * 2 + 1024);
            AllocatePtr(d_active, pointCapacity, "alloc active");
            AllocatePtr(d_scale, pointCapacity, "alloc scale");
            AllocatePtr(d_viewCos, pointCapacity, "alloc view cos");
            AllocatePtr(d_projX, pointCapacity, "alloc proj x");
            AllocatePtr(d_projY, pointCapacity, "alloc proj y");
            AllocatePtr(d_projXR, pointCapacity, "alloc proj xr");
            AllocatePtr(d_mpDesc, pointCapacity * kDescriptorSize, "alloc mp desc");
            AllocatePtr(d_bestLevel, pointCapacity, "alloc best level");
            AllocatePtr(d_bestLevel2, pointCapacity, "alloc best level2");
            AllocatePtr(d_bestDist, pointCapacity, "alloc best dist");
            AllocatePtr(d_bestDist2, pointCapacity, "alloc best dist2");
            AllocatePtr(d_bestIdx, pointCapacity, "alloc best idx");
        }

        if(numFeatures > featureCapacity)
        {
            FreePtr(d_keyX);
            FreePtr(d_keyY);
            FreePtr(d_keyOctave);
            FreePtr(d_frameDesc);
            FreePtr(d_mvuRight);
            FreePtr(d_frameOccupied);

            featureCapacity = std::max(numFeatures, featureCapacity * 2 + 1024);
            AllocatePtr(d_keyX, featureCapacity, "alloc key x");
            AllocatePtr(d_keyY, featureCapacity, "alloc key y");
            AllocatePtr(d_keyOctave, featureCapacity, "alloc key octave");
            AllocatePtr(d_frameDesc, featureCapacity * kDescriptorSize, "alloc frame desc");
            AllocatePtr(d_mvuRight, featureCapacity, "alloc right coord");
            AllocatePtr(d_frameOccupied, featureCapacity, "alloc occupied");
        }

        if(!d_grid)
        {
            AllocatePtr(d_grid, kGridCols * kGridRows * kKeypointsPerCell, "alloc grid");
            AllocatePtr(d_gridSize, kGridCols * kGridRows, "alloc grid size");
        }

        if(numLevels > levelCapacity)
        {
            FreePtr(d_scaleFactors);
            levelCapacity = std::max(numLevels, levelCapacity * 2 + 8);
            AllocatePtr(d_scaleFactors, levelCapacity, "alloc scale factors");
        }
    }

    void Free()
    {
        FreePtr(d_active);
        FreePtr(d_scale);
        FreePtr(d_viewCos);
        FreePtr(d_projX);
        FreePtr(d_projY);
        FreePtr(d_projXR);
        FreePtr(d_mpDesc);
        FreePtr(d_keyX);
        FreePtr(d_keyY);
        FreePtr(d_keyOctave);
        FreePtr(d_frameDesc);
        FreePtr(d_mvuRight);
        FreePtr(d_frameOccupied);
        FreePtr(d_grid);
        FreePtr(d_gridSize);
        FreePtr(d_scaleFactors);
        FreePtr(d_bestLevel);
        FreePtr(d_bestLevel2);
        FreePtr(d_bestDist);
        FreePtr(d_bestDist2);
        FreePtr(d_bestIdx);
        pointCapacity = featureCapacity = levelCapacity = 0;
    }
};

DeviceBuffers gBuffers;

} // namespace

namespace ORB_SLAM3
{

bool AntSlamCudaLocalMatcher::Enabled()
{
    return EnvBool("ANTSLAM_CUDA_LOCAL_MATCHER", false);
}

bool AntSlamCudaLocalMatcher::Match(Frame &F,
                                    const std::vector<MapPoint*> &vpMapPoints,
                                    const float th,
                                    const bool bFarPoints,
                                    const float thFarPoints,
                                    std::vector<int> &bestLevel,
                                    std::vector<int> &bestLevel2,
                                    std::vector<int> &bestDist,
                                    std::vector<int> &bestDist2,
                                    std::vector<int> &bestIdx,
                                    std::vector<int> &bestLevelR,
                                    std::vector<int> &bestLevelR2,
                                    std::vector<int> &bestDistR,
                                    std::vector<int> &bestDistR2,
                                    std::vector<int> &bestIdxR)
{
    if(!Enabled() || F.Nleft != -1 || vpMapPoints.empty() || F.N <= 0)
        return false;

    const int numPoints = static_cast<int>(vpMapPoints.size());
    const int numFeatures = F.N;
    const int numLevels = F.mnScaleLevels;

    try
    {
        gBuffers.Ensure(numPoints, numFeatures, numLevels);

        std::vector<unsigned char> active(numPoints, 0);
        std::vector<int> scale(numPoints, -1);
        std::vector<float> viewCos(numPoints, 0.0f);
        std::vector<float> projX(numPoints, 0.0f);
        std::vector<float> projY(numPoints, 0.0f);
        std::vector<float> projXR(numPoints, 0.0f);
        std::vector<unsigned char> mpDesc(numPoints * kDescriptorSize, 0);

        for(int i = 0; i < numPoints; ++i)
        {
            MapPoint *pMP = vpMapPoints[i];
            if(!pMP || !pMP->mbTrackInView || pMP->isBad())
                continue;
            if(bFarPoints && pMP->mTrackDepth > thFarPoints)
                continue;
            if(pMP->mnTrackScaleLevel < 0 || pMP->mnTrackScaleLevel >= numLevels)
                continue;

            active[i] = 1;
            scale[i] = pMP->mnTrackScaleLevel;
            viewCos[i] = pMP->mTrackViewCos;
            projX[i] = pMP->mTrackProjX;
            projY[i] = pMP->mTrackProjY;
            projXR[i] = pMP->mTrackProjXR;
            const cv::Mat descriptor = pMP->GetDescriptor();
            if(descriptor.cols >= kDescriptorSize)
                std::memcpy(&mpDesc[i * kDescriptorSize], descriptor.ptr<unsigned char>(0), kDescriptorSize);
        }

        std::vector<float> keyX(numFeatures, 0.0f);
        std::vector<float> keyY(numFeatures, 0.0f);
        std::vector<int> keyOctave(numFeatures, 0);
        for(int i = 0; i < numFeatures && i < static_cast<int>(F.mvKeysUn.size()); ++i)
        {
            keyX[i] = F.mvKeysUn[i].pt.x;
            keyY[i] = F.mvKeysUn[i].pt.y;
            keyOctave[i] = F.mvKeysUn[i].octave;
        }

        std::vector<unsigned char> frameDesc(numFeatures * kDescriptorSize, 0);
        if(F.mDescriptors.rows >= numFeatures)
        {
            for(int i = 0; i < numFeatures; ++i)
                std::memcpy(&frameDesc[i * kDescriptorSize], F.mDescriptors.ptr<unsigned char>(i), kDescriptorSize);
        }

        std::vector<float> mvuRight(numFeatures, -1.0f);
        for(int i = 0; i < numFeatures && i < static_cast<int>(F.mvuRight.size()); ++i)
            mvuRight[i] = F.mvuRight[i];

        std::vector<unsigned char> occupied(numFeatures, 0);
        for(int i = 0; i < numFeatures && i < static_cast<int>(F.mvpMapPoints.size()); ++i)
        {
            if(F.mvpMapPoints[i] && F.mvpMapPoints[i]->Observations() > 0)
                occupied[i] = 1;
        }

        std::vector<int> grid(kGridCols * kGridRows * kKeypointsPerCell, -1);
        std::vector<int> gridSize(kGridCols * kGridRows, 0);
        for(int ix = 0; ix < kGridCols; ++ix)
        {
            for(int iy = 0; iy < kGridRows; ++iy)
            {
                const std::vector<size_t> &cell = F.mGrid[ix][iy];
                const int count = std::min(static_cast<int>(cell.size()), kKeypointsPerCell);
                gridSize[ix * kGridRows + iy] = count;
                for(int j = 0; j < count; ++j)
                    grid[(ix * kGridRows + iy) * kKeypointsPerCell + j] = static_cast<int>(cell[j]);
            }
        }

        CheckCuda(cudaMemcpy(gBuffers.d_active, active.data(), numPoints * sizeof(unsigned char), cudaMemcpyHostToDevice), "copy active");
        CheckCuda(cudaMemcpy(gBuffers.d_scale, scale.data(), numPoints * sizeof(int), cudaMemcpyHostToDevice), "copy scale");
        CheckCuda(cudaMemcpy(gBuffers.d_viewCos, viewCos.data(), numPoints * sizeof(float), cudaMemcpyHostToDevice), "copy view cos");
        CheckCuda(cudaMemcpy(gBuffers.d_projX, projX.data(), numPoints * sizeof(float), cudaMemcpyHostToDevice), "copy proj x");
        CheckCuda(cudaMemcpy(gBuffers.d_projY, projY.data(), numPoints * sizeof(float), cudaMemcpyHostToDevice), "copy proj y");
        CheckCuda(cudaMemcpy(gBuffers.d_projXR, projXR.data(), numPoints * sizeof(float), cudaMemcpyHostToDevice), "copy proj xr");
        CheckCuda(cudaMemcpy(gBuffers.d_mpDesc, mpDesc.data(), numPoints * kDescriptorSize * sizeof(unsigned char), cudaMemcpyHostToDevice), "copy mp desc");
        CheckCuda(cudaMemcpy(gBuffers.d_keyX, keyX.data(), numFeatures * sizeof(float), cudaMemcpyHostToDevice), "copy key x");
        CheckCuda(cudaMemcpy(gBuffers.d_keyY, keyY.data(), numFeatures * sizeof(float), cudaMemcpyHostToDevice), "copy key y");
        CheckCuda(cudaMemcpy(gBuffers.d_keyOctave, keyOctave.data(), numFeatures * sizeof(int), cudaMemcpyHostToDevice), "copy key octave");
        CheckCuda(cudaMemcpy(gBuffers.d_frameDesc, frameDesc.data(), numFeatures * kDescriptorSize * sizeof(unsigned char), cudaMemcpyHostToDevice), "copy frame desc");
        CheckCuda(cudaMemcpy(gBuffers.d_mvuRight, mvuRight.data(), numFeatures * sizeof(float), cudaMemcpyHostToDevice), "copy mvu right");
        CheckCuda(cudaMemcpy(gBuffers.d_frameOccupied, occupied.data(), numFeatures * sizeof(unsigned char), cudaMemcpyHostToDevice), "copy occupied");
        CheckCuda(cudaMemcpy(gBuffers.d_grid, grid.data(), grid.size() * sizeof(int), cudaMemcpyHostToDevice), "copy grid");
        CheckCuda(cudaMemcpy(gBuffers.d_gridSize, gridSize.data(), gridSize.size() * sizeof(int), cudaMemcpyHostToDevice), "copy grid size");
        CheckCuda(cudaMemcpy(gBuffers.d_scaleFactors, F.mvScaleFactors.data(), numLevels * sizeof(float), cudaMemcpyHostToDevice), "copy scale factors");

        AntSlamLaunchLocalSearchKernel(gBuffers.d_active, gBuffers.d_scale, gBuffers.d_viewCos,
                                       gBuffers.d_projX, gBuffers.d_projY, gBuffers.d_projXR,
                                       gBuffers.d_mpDesc, gBuffers.d_keyX, gBuffers.d_keyY,
                                       gBuffers.d_keyOctave, gBuffers.d_frameDesc, gBuffers.d_mvuRight,
                                       gBuffers.d_frameOccupied, gBuffers.d_grid, gBuffers.d_gridSize,
                                       gBuffers.d_scaleFactors, numPoints, numFeatures, numLevels,
                                       F.mnMinX, F.mnMinY, F.mfGridElementWidthInv, F.mfGridElementHeightInv,
                                       th, gBuffers.d_bestLevel, gBuffers.d_bestLevel2,
                                       gBuffers.d_bestDist, gBuffers.d_bestDist2, gBuffers.d_bestIdx);
        CheckCuda(cudaDeviceSynchronize(), "local search kernel");

        bestLevel.assign(numPoints, -1);
        bestLevel2.assign(numPoints, -1);
        bestDist.assign(numPoints, 256);
        bestDist2.assign(numPoints, 256);
        bestIdx.assign(numPoints, -1);
        CheckCuda(cudaMemcpy(bestLevel.data(), gBuffers.d_bestLevel, numPoints * sizeof(int), cudaMemcpyDeviceToHost), "read best level");
        CheckCuda(cudaMemcpy(bestLevel2.data(), gBuffers.d_bestLevel2, numPoints * sizeof(int), cudaMemcpyDeviceToHost), "read best level2");
        CheckCuda(cudaMemcpy(bestDist.data(), gBuffers.d_bestDist, numPoints * sizeof(int), cudaMemcpyDeviceToHost), "read best dist");
        CheckCuda(cudaMemcpy(bestDist2.data(), gBuffers.d_bestDist2, numPoints * sizeof(int), cudaMemcpyDeviceToHost), "read best dist2");
        CheckCuda(cudaMemcpy(bestIdx.data(), gBuffers.d_bestIdx, numPoints * sizeof(int), cudaMemcpyDeviceToHost), "read best idx");

        bestLevelR.assign(numPoints, -1);
        bestLevelR2.assign(numPoints, -1);
        bestDistR.assign(numPoints, 256);
        bestDistR2.assign(numPoints, 256);
        bestIdxR.assign(numPoints, -1);
        return true;
    }
    catch(const std::exception&)
    {
        return false;
    }
}

void AntSlamCudaLocalMatcher::Shutdown()
{
    gBuffers.Free();
}

} // namespace ORB_SLAM3

#endif
