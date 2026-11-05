#ifndef ORB_SLAM3_CUDA_SEMANTIC_MASK_KERNELS_HPP
#define ORB_SLAM3_CUDA_SEMANTIC_MASK_KERNELS_HPP

#include <cuda_runtime_api.h>

namespace ORB_SLAM3
{
namespace cuda_semantic
{

enum TensorValueType
{
    kTensorFloat32 = 0,
    kTensorFloat16 = 1
};

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
    cudaStream_t stream);

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
    cudaStream_t stream);

} // namespace cuda_semantic
} // namespace ORB_SLAM3

#endif
