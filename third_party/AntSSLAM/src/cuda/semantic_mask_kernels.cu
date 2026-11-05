#include "cuda/semantic_mask_kernels.hpp"

#include <cuda_fp16.h>
#include <math_constants.h>

namespace ORB_SLAM3
{
namespace cuda_semantic
{
namespace
{

__device__ __forceinline__ float ReadValue(const void *data, int type, size_t index)
{
    if(type == kTensorFloat16)
        return __half2float(static_cast<const __half*>(data)[index]);
    return static_cast<const float*>(data)[index];
}

__device__ __forceinline__ float FastSigmoid(float x)
{
    if(x >= 0.0f)
    {
        const float z = __expf(-x);
        return 1.0f / (1.0f + z);
    }

    const float z = __expf(x);
    return z / (1.0f + z);
}

__device__ __forceinline__ void WriteTensorValue(void *data, int type, size_t index, float value)
{
    if(type == kTensorFloat16)
        static_cast<__half*>(data)[index] = __float2half_rn(value);
    else
        static_cast<float*>(data)[index] = value;
}

__global__ void PrepareBgrInputKernel(
    const unsigned char *imageBgr,
    int imageWidth,
    int imageHeight,
    size_t imageStepBytes,
    void *inputTensor,
    int inputType,
    int inputWidth,
    int inputHeight,
    int projectionWidth,
    int projectionHeight)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = inputWidth * inputHeight;
    if(index >= total)
        return;

    const int x = index % inputWidth;
    const int y = index / inputWidth;
    const size_t plane = static_cast<size_t>(inputWidth) * static_cast<size_t>(inputHeight);

    float b = 0.0f;
    float g = 0.0f;
    float r = 0.0f;

    if(x < projectionWidth && y < projectionHeight)
    {
        const float srcX = (static_cast<float>(x) + 0.5f) *
                               static_cast<float>(imageWidth) / static_cast<float>(projectionWidth) -
                           0.5f;
        const float srcY = (static_cast<float>(y) + 0.5f) *
                               static_cast<float>(imageHeight) / static_cast<float>(projectionHeight) -
                           0.5f;
        const int x0 = max(0, min(imageWidth - 1, static_cast<int>(floorf(srcX))));
        const int y0 = max(0, min(imageHeight - 1, static_cast<int>(floorf(srcY))));
        const int x1 = min(imageWidth - 1, x0 + 1);
        const int y1 = min(imageHeight - 1, y0 + 1);
        const float wx = min(1.0f, max(0.0f, srcX - static_cast<float>(x0)));
        const float wy = min(1.0f, max(0.0f, srcY - static_cast<float>(y0)));

        const unsigned char *row0 = imageBgr + static_cast<size_t>(y0) * imageStepBytes;
        const unsigned char *row1 = imageBgr + static_cast<size_t>(y1) * imageStepBytes;
        const unsigned char *p00 = row0 + static_cast<size_t>(x0) * 3;
        const unsigned char *p01 = row0 + static_cast<size_t>(x1) * 3;
        const unsigned char *p10 = row1 + static_cast<size_t>(x0) * 3;
        const unsigned char *p11 = row1 + static_cast<size_t>(x1) * 3;
        const float w00 = (1.0f - wx) * (1.0f - wy);
        const float w01 = wx * (1.0f - wy);
        const float w10 = (1.0f - wx) * wy;
        const float w11 = wx * wy;
        b = (w00 * p00[0] + w01 * p01[0] + w10 * p10[0] + w11 * p11[0]) * (1.0f / 255.0f);
        g = (w00 * p00[1] + w01 * p01[1] + w10 * p10[1] + w11 * p11[1]) * (1.0f / 255.0f);
        r = (w00 * p00[2] + w01 * p01[2] + w10 * p10[2] + w11 * p11[2]) * (1.0f / 255.0f);
    }

    WriteTensorValue(inputTensor, inputType, index, r);
    WriteTensorValue(inputTensor, inputType, plane + index, g);
    WriteTensorValue(inputTensor, inputType, 2 * plane + index, b);
}

__device__ __forceinline__ bool DynamicAtImagePixel(
    int x,
    int y,
    const unsigned char *dynamicSmall,
    int maskHeight,
    int maskWidth,
    int inputWidth,
    int inputHeight,
    int projectionWidth,
    int projectionHeight,
    int imageWidth,
    int imageHeight)
{
    if(x < 0 || x >= imageWidth || y < 0 || y >= imageHeight)
        return false;

    const int validX = min(projectionWidth - 1,
                           static_cast<int>((static_cast<long long>(x) * projectionWidth) / imageWidth));
    const int validY = min(projectionHeight - 1,
                           static_cast<int>((static_cast<long long>(y) * projectionHeight) / imageHeight));
    const int maskX = min(maskWidth - 1,
                          static_cast<int>((static_cast<long long>(validX) * maskWidth) / inputWidth));
    const int maskY = min(maskHeight - 1,
                          static_cast<int>((static_cast<long long>(validY) * maskHeight) / inputHeight));
    return dynamicSmall[maskY * maskWidth + maskX] != 0;
}

__global__ void ClassifyQueriesKernel(
    const void *classLogits,
    int classType,
    int queryCount,
    int classCount,
    float threshold,
    int *queryClass,
    float *queryScore,
    unsigned char *queryKeep)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if(q >= queryCount)
        return;

    const size_t base = static_cast<size_t>(q) * static_cast<size_t>(classCount);
    float maxLogit = ReadValue(classLogits, classType, base);
    int maxClass = 0;
    for(int c = 1; c < classCount; ++c)
    {
        const float value = ReadValue(classLogits, classType, base + c);
        if(value > maxLogit)
        {
            maxLogit = value;
            maxClass = c;
        }
    }

    float expSum = 0.0f;
    for(int c = 0; c < classCount; ++c)
        expSum += __expf(ReadValue(classLogits, classType, base + c) - maxLogit);

    const float score = expSum > 0.0f ? 1.0f / expSum : 0.0f;
    queryClass[q] = maxClass;
    queryScore[q] = score;
    queryKeep[q] = static_cast<unsigned char>((maxClass != classCount - 1) && (score > threshold));
}

__global__ void BestQueryKernel(
    const void *maskLogits,
    int maskType,
    int queryCount,
    int area,
    const float *queryScore,
    const unsigned char *queryKeep,
    int *bestQuery)
{
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if(pixel >= area)
        return;

    float bestScore = -CUDART_INF_F;
    int best = -1;
    for(int q = 0; q < queryCount; ++q)
    {
        if(!queryKeep[q])
            continue;
        const float maskScore = queryScore[q] *
            FastSigmoid(ReadValue(maskLogits, maskType, static_cast<size_t>(q) * area + pixel));
        if(maskScore > bestScore)
        {
            bestScore = maskScore;
            best = q;
        }
    }
    bestQuery[pixel] = best;
}

__global__ void QueryAreasKernel(
    const void *maskLogits,
    int maskType,
    int queryCount,
    int area,
    const unsigned char *queryKeep,
    const int *bestQuery,
    int *originalArea,
    int *winnerArea,
    int *finalArea)
{
    const int q = blockIdx.x;
    if(q >= queryCount)
        return;

    __shared__ int originalShared[256];
    __shared__ int winnerShared[256];
    __shared__ int finalShared[256];

    int original = 0;
    int winner = 0;
    int final = 0;

    if(queryKeep[q])
    {
        for(int pixel = threadIdx.x; pixel < area; pixel += blockDim.x)
        {
            const bool originalMask =
                ReadValue(maskLogits, maskType, static_cast<size_t>(q) * area + pixel) >= 0.0f;
            const bool winnerMask = bestQuery[pixel] == q;
            original += originalMask ? 1 : 0;
            winner += winnerMask ? 1 : 0;
            final += (originalMask && winnerMask) ? 1 : 0;
        }
    }

    originalShared[threadIdx.x] = original;
    winnerShared[threadIdx.x] = winner;
    finalShared[threadIdx.x] = final;
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if(threadIdx.x < stride)
        {
            originalShared[threadIdx.x] += originalShared[threadIdx.x + stride];
            winnerShared[threadIdx.x] += winnerShared[threadIdx.x + stride];
            finalShared[threadIdx.x] += finalShared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0)
    {
        originalArea[q] = originalShared[0];
        winnerArea[q] = winnerShared[0];
        finalArea[q] = finalShared[0];
    }
}

__global__ void FinalizeQueriesKernel(
    int queryCount,
    const int *queryClass,
    const unsigned char *dynamicClassMask,
    float overlapThreshold,
    const int *originalArea,
    const int *winnerArea,
    const int *finalArea,
    unsigned char *queryAccepted)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if(q >= queryCount)
        return;

    const int original = originalArea[q];
    const int winner = winnerArea[q];
    const int final = finalArea[q];
    const int classId = queryClass[q];
    const bool valid = original > 0 && winner > 0 && final > 0 &&
                       dynamicClassMask[classId] &&
                       (static_cast<float>(winner) / static_cast<float>(original) >= overlapThreshold);
    queryAccepted[q] = static_cast<unsigned char>(valid);
}

__global__ void BuildDynamicSmallKernel(
    const void *maskLogits,
    int maskType,
    int area,
    const int *bestQuery,
    const unsigned char *queryAccepted,
    unsigned char *dynamicSmall)
{
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if(pixel >= area)
        return;

    const int query = bestQuery[pixel];
    bool dynamic = false;
    if(query >= 0 && queryAccepted[query])
        dynamic = ReadValue(maskLogits, maskType, static_cast<size_t>(query) * area + pixel) >= 0.0f;
    dynamicSmall[pixel] = dynamic ? 255 : 0;
}

__global__ void ProjectFeatureMaskKernel(
    const unsigned char *dynamicSmall,
    int maskHeight,
    int maskWidth,
    int inputWidth,
    int inputHeight,
    int projectionWidth,
    int projectionHeight,
    int imageWidth,
    int imageHeight,
    int dilatePixels,
    unsigned char *featureMask)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = imageWidth * imageHeight;
    if(index >= total)
        return;

    const int x = index % imageWidth;
    const int y = index / imageWidth;
    bool reject = false;

    if(dilatePixels <= 0)
    {
        reject = DynamicAtImagePixel(x, y, dynamicSmall, maskHeight, maskWidth,
                                     inputWidth, inputHeight, projectionWidth, projectionHeight,
                                     imageWidth, imageHeight);
    }
    else
    {
        const int r = dilatePixels;
        const int r2 = r * r;
        for(int dy = -r; dy <= r && !reject; ++dy)
        {
            for(int dx = -r; dx <= r; ++dx)
            {
                if(dx * dx + dy * dy > r2)
                    continue;
                if(DynamicAtImagePixel(x + dx, y + dy, dynamicSmall, maskHeight, maskWidth,
                                       inputWidth, inputHeight, projectionWidth, projectionHeight,
                                       imageWidth, imageHeight))
                {
                    reject = true;
                    break;
                }
            }
        }
    }

    featureMask[index] = reject ? 0 : 255;
}

} // namespace

cudaError_t PrepareTokenMaskInputFromBgr(
    const unsigned char *imageBgr,
    int imageWidth,
    int imageHeight,
    size_t imageStepBytes,
    void *inputTensor,
    int inputType,
    int inputWidth,
    int inputHeight,
    int projectionWidth,
    int projectionHeight,
    cudaStream_t stream)
{
    if(!imageBgr || !inputTensor || imageWidth <= 0 || imageHeight <= 0 ||
       imageStepBytes < static_cast<size_t>(imageWidth) * 3 || inputWidth <= 0 ||
       inputHeight <= 0 || projectionWidth <= 0 || projectionHeight <= 0 ||
       projectionWidth > inputWidth || projectionHeight > inputHeight)
    {
        return cudaErrorInvalidValue;
    }

    const int total = inputWidth * inputHeight;
    const int threads = 256;
    PrepareBgrInputKernel<<<(total + threads - 1) / threads, threads, 0, stream>>>(
        imageBgr, imageWidth, imageHeight, imageStepBytes, inputTensor, inputType,
        inputWidth, inputHeight, projectionWidth, projectionHeight);
    return cudaGetLastError();
}

cudaError_t DecodeTokenMaskFeatureMask(
    const void *maskLogits,
    int maskType,
    const void *classLogits,
    int classType,
    int queryCount,
    int maskHeight,
    int maskWidth,
    int classCount,
    const unsigned char *dynamicClassMask,
    float scoreThreshold,
    float overlapThreshold,
    int inputWidth,
    int inputHeight,
    int projectionWidth,
    int projectionHeight,
    int imageWidth,
    int imageHeight,
    int dilatePixels,
    int *queryClass,
    float *queryScore,
    unsigned char *queryKeep,
    int *bestQuery,
    int *originalArea,
    int *winnerArea,
    int *finalArea,
    unsigned char *queryAccepted,
    unsigned char *dynamicSmall,
    unsigned char *featureMask,
    cudaStream_t stream)
{
    if(!maskLogits || !classLogits || !dynamicClassMask || !queryClass || !queryScore ||
       !queryKeep || !bestQuery || !originalArea || !winnerArea || !finalArea ||
       !queryAccepted || !dynamicSmall || !featureMask || queryCount <= 0 || maskHeight <= 0 ||
       maskWidth <= 0 || classCount <= 1 || inputWidth <= 0 || inputHeight <= 0 ||
       projectionWidth <= 0 || projectionHeight <= 0 || imageWidth <= 0 || imageHeight <= 0)
    {
        return cudaErrorInvalidValue;
    }

    const int area = maskHeight * maskWidth;
    const int queryThreads = 128;
    const int pixelThreads = 256;
    ClassifyQueriesKernel<<<(queryCount + queryThreads - 1) / queryThreads, queryThreads, 0, stream>>>(
        classLogits, classType, queryCount, classCount, scoreThreshold,
        queryClass, queryScore, queryKeep);
    BestQueryKernel<<<(area + pixelThreads - 1) / pixelThreads, pixelThreads, 0, stream>>>(
        maskLogits, maskType, queryCount, area, queryScore, queryKeep, bestQuery);
    QueryAreasKernel<<<queryCount, 256, 0, stream>>>(
        maskLogits, maskType, queryCount, area, queryKeep, bestQuery,
        originalArea, winnerArea, finalArea);
    FinalizeQueriesKernel<<<(queryCount + queryThreads - 1) / queryThreads, queryThreads, 0, stream>>>(
        queryCount, queryClass, dynamicClassMask, overlapThreshold,
        originalArea, winnerArea, finalArea, queryAccepted);
    BuildDynamicSmallKernel<<<(area + pixelThreads - 1) / pixelThreads, pixelThreads, 0, stream>>>(
        maskLogits, maskType, area, bestQuery, queryAccepted, dynamicSmall);
    ProjectFeatureMaskKernel<<<(imageWidth * imageHeight + pixelThreads - 1) / pixelThreads,
                               pixelThreads, 0, stream>>>(
        dynamicSmall, maskHeight, maskWidth,
        inputWidth, inputHeight, projectionWidth, projectionHeight,
        imageWidth, imageHeight, dilatePixels, featureMask);

    return cudaGetLastError();
}

} // namespace cuda_semantic
} // namespace ORB_SLAM3
