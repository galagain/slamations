#include <cuda_runtime.h>
#include <stdint.h>

namespace
{

constexpr int kGridCols = 64;
constexpr int kGridRows = 48;
constexpr int kKeypointsPerCell = 20;
constexpr int kDescriptorSize = 32;
constexpr int kThHigh = 100;

__device__ int DescriptorDistance(const unsigned char *a, const unsigned char *b)
{
    int dist = 0;
    #pragma unroll
    for(int i = 0; i < 8; ++i)
    {
        const unsigned int av = static_cast<unsigned int>(a[4*i]) |
                                (static_cast<unsigned int>(a[4*i+1]) << 8) |
                                (static_cast<unsigned int>(a[4*i+2]) << 16) |
                                (static_cast<unsigned int>(a[4*i+3]) << 24);
        const unsigned int bv = static_cast<unsigned int>(b[4*i]) |
                                (static_cast<unsigned int>(b[4*i+1]) << 8) |
                                (static_cast<unsigned int>(b[4*i+2]) << 16) |
                                (static_cast<unsigned int>(b[4*i+3]) << 24);
        dist += __popc(av ^ bv);
    }
    return dist;
}

__global__ void LocalSearchKernel(
    const unsigned char *active,
    const int *scale,
    const float *viewCos,
    const float *projX,
    const float *projY,
    const float *projXR,
    const unsigned char *mpDesc,
    const float *keyX,
    const float *keyY,
    const int *keyOctave,
    const unsigned char *frameDesc,
    const float *mvuRight,
    const unsigned char *frameOccupied,
    const int *grid,
    const int *gridSize,
    const float *scaleFactors,
    int numPoints,
    int numFeatures,
    int numLevels,
    float minX,
    float minY,
    float gridElementWidthInv,
    float gridElementHeightInv,
    float th,
    int *bestLevel,
    int *bestLevel2,
    int *bestDist,
    int *bestDist2,
    int *bestIdx)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numPoints)
        return;

    bestDist[idx] = 256;
    bestDist2[idx] = 256;
    bestLevel[idx] = -1;
    bestLevel2[idx] = -1;
    bestIdx[idx] = -1;

    if(!active[idx])
        return;

    const int predictedLevel = scale[idx];
    if(predictedLevel < 0 || predictedLevel >= numLevels)
        return;

    const float x = projX[idx];
    const float y = projY[idx];
    float r = (viewCos[idx] > 0.998f) ? 2.5f : 4.0f;
    r *= th;
    const float radius = r * scaleFactors[predictedLevel];

    const int minLevel = predictedLevel - 1;
    const int maxLevel = predictedLevel;
    const bool checkLevels = (minLevel > 0) || (maxLevel >= 0);

    const int minCellX = max(0, static_cast<int>(floorf((x - minX - radius) * gridElementWidthInv)));
    if(minCellX >= kGridCols)
        return;
    const int maxCellX = min(kGridCols - 1, static_cast<int>(ceilf((x - minX + radius) * gridElementWidthInv)));
    if(maxCellX < 0)
        return;
    const int minCellY = max(0, static_cast<int>(floorf((y - minY - radius) * gridElementHeightInv)));
    if(minCellY >= kGridRows)
        return;
    const int maxCellY = min(kGridRows - 1, static_cast<int>(ceilf((y - minY + radius) * gridElementHeightInv)));
    if(maxCellY < 0)
        return;

    const unsigned char *mpDescriptor = mpDesc + idx * kDescriptorSize;
    int localBestDist = 256;
    int localBestDist2 = 256;
    int localBestLevel = -1;
    int localBestLevel2 = -1;
    int localBestIdx = -1;

    for(int ix = minCellX; ix <= maxCellX; ++ix)
    {
        for(int iy = minCellY; iy <= maxCellY; ++iy)
        {
            const int cellOffset = ix * kGridRows + iy;
            const int count = gridSize[cellOffset];
            for(int j = 0; j < count; ++j)
            {
                const int featureIdx = grid[cellOffset * kKeypointsPerCell + j];
                if(featureIdx < 0 || featureIdx >= numFeatures)
                    continue;
                if(frameOccupied[featureIdx])
                    continue;

                const int octave = keyOctave[featureIdx];
                if(checkLevels)
                {
                    if(octave < minLevel)
                        continue;
                    if(maxLevel >= 0 && octave > maxLevel)
                        continue;
                }

                const float dx = keyX[featureIdx] - x;
                const float dy = keyY[featureIdx] - y;
                if(fabsf(dx) >= radius || fabsf(dy) >= radius)
                    continue;

                if(mvuRight[featureIdx] > 0.0f)
                {
                    const float er = fabsf(projXR[idx] - mvuRight[featureIdx]);
                    if(er > radius)
                        continue;
                }

                const int dist = DescriptorDistance(mpDescriptor, frameDesc + featureIdx * kDescriptorSize);
                if(dist < localBestDist)
                {
                    localBestDist2 = localBestDist;
                    localBestLevel2 = localBestLevel;
                    localBestDist = dist;
                    localBestLevel = octave;
                    localBestIdx = featureIdx;
                }
                else if(dist < localBestDist2)
                {
                    localBestDist2 = dist;
                    localBestLevel2 = octave;
                }
            }
        }
    }

    if(localBestDist <= kThHigh)
    {
        bestDist[idx] = localBestDist;
        bestDist2[idx] = localBestDist2;
        bestLevel[idx] = localBestLevel;
        bestLevel2[idx] = localBestLevel2;
        bestIdx[idx] = localBestIdx;
    }
}

} // namespace

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
    int *d_bestIdx)
{
    const int blockSize = 256;
    const int gridSize = (numPoints + blockSize - 1) / blockSize;
    LocalSearchKernel<<<gridSize, blockSize>>>(
        d_active, d_scale, d_viewCos, d_projX, d_projY, d_projXR, d_mpDesc,
        d_keyX, d_keyY, d_keyOctave, d_frameDesc, d_mvuRight, d_frameOccupied,
        d_grid, d_gridSize, d_scaleFactors, numPoints, numFeatures, numLevels,
        minX, minY, gridElementWidthInv, gridElementHeightInv, th,
        d_bestLevel, d_bestLevel2, d_bestDist, d_bestDist2, d_bestIdx);
}
