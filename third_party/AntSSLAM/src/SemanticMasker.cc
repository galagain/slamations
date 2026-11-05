#include "SemanticMasker.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/types.h>
#include <thread>
#include <vector>

#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#ifdef LITESEMSLAM_USE_TENSORRT_SEMANTICS
#include "cuda/semantic_mask_kernels.hpp"

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#endif

namespace ORB_SLAM3
{

namespace
{
std::string JoinPath(const std::string &base, const std::string &leaf)
{
    if(base.empty())
        return leaf;
    if(base[base.size() - 1] == '/')
        return base + leaf;
    return base + "/" + leaf;
}

std::string ToLower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    return value;
}

std::string Trim(const std::string &value)
{
    const size_t begin = value.find_first_not_of(" \t\r\n");
    if(begin == std::string::npos)
        return std::string();
    const size_t end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

std::vector<std::string> SplitLabels(const std::string &labels)
{
    std::string normalized = labels;
    std::replace(normalized.begin(), normalized.end(), ';', ',');

    std::vector<std::string> tokens;
    std::stringstream stream(normalized);
    std::string token;
    while(std::getline(stream, token, ','))
    {
        token = ToLower(Trim(token));
        if(!token.empty())
            tokens.push_back(token);
    }
    return tokens;
}

bool LooksLikeInteger(const std::string &value)
{
    if(value.empty())
        return false;
    size_t start = value[0] == '-' ? 1 : 0;
    if(start == value.size())
        return false;
    for(size_t i = start; i < value.size(); ++i)
    {
        if(!std::isdigit(static_cast<unsigned char>(value[i])))
            return false;
    }
    return true;
}

bool ContainsToken(const std::string &value, const std::string &token)
{
    return ToLower(value).find(ToLower(token)) != std::string::npos;
}

bool EndsWithToken(const std::string &value, const std::string &token)
{
    if(value.size() < token.size())
        return false;
    return ToLower(value.substr(value.size() - token.size())) == ToLower(token);
}

bool SemanticTraceEnabled()
{
    const char *value = std::getenv("LITESEMSLAM_SEMANTIC_TRACE");
    return value && *value && std::atoi(value) != 0;
}

double MsSince(const std::chrono::steady_clock::time_point &start,
               const std::chrono::steady_clock::time_point &end)
{
    return std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(end - start).count();
}

float Sigmoid(float x)
{
    if(x >= 0.0f)
    {
        const float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }

    const float z = std::exp(x);
    return z / (1.0f + z);
}

std::set<int> ParseDynamicLabels(const std::string &labels,
                                 const std::map<std::string, int> &nameToId,
                                 const std::set<int> &defaultIds)
{
    if(labels.empty())
        return defaultIds;

    std::set<int> ids;
    for(const std::string &token : SplitLabels(labels))
    {
        if(LooksLikeInteger(token))
        {
            ids.insert(std::atoi(token.c_str()));
            continue;
        }

        std::map<std::string, int>::const_iterator it = nameToId.find(token);
        if(it != nameToId.end())
            ids.insert(it->second);
        else
            std::cerr << "SemanticMasker: ignoring unknown dynamic label '" << token << "'" << std::endl;
    }

    return ids.empty() ? defaultIds : ids;
}

void DynamicToFeatureMask(const cv::Mat &dynamicMask, int dilatePixels, cv::Mat &featureMask)
{
    cv::Mat dynamic = dynamicMask;
    cv::Mat dilated;
    if(dilatePixels > 0 && cv::countNonZero(dynamicMask) > 0)
    {
        const int kernelSize = 2 * dilatePixels + 1;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernelSize, kernelSize));
        cv::dilate(dynamicMask, dilated, kernel);
        dynamic = dilated;
    }

    featureMask.create(dynamic.rows, dynamic.cols, CV_8UC1);
    featureMask.setTo(cv::Scalar(255));
    featureMask.setTo(cv::Scalar(0), dynamic != 0);
}

cv::Mat PrepareImageForBackend(const cv::Mat &image, bool inputIsRgb)
{
    cv::Mat bgr;
    if(image.empty())
        return bgr;

    if(image.channels() == 1)
        cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
    else if(image.channels() == 3)
    {
        if(inputIsRgb)
            cv::cvtColor(image, bgr, cv::COLOR_RGB2BGR);
        else
            bgr = image.clone();
    }
    else if(image.channels() == 4)
    {
        if(inputIsRgb)
            cv::cvtColor(image, bgr, cv::COLOR_RGBA2BGR);
        else
            cv::cvtColor(image, bgr, cv::COLOR_BGRA2BGR);
    }

    return bgr;
}

uint64_t MatChecksum(const cv::Mat &mat)
{
    uint64_t hash = 1469598103934665603ULL;
    const uint64_t prime = 1099511628211ULL;

    hash ^= static_cast<uint64_t>(mat.rows);
    hash *= prime;
    hash ^= static_cast<uint64_t>(mat.cols);
    hash *= prime;
    hash ^= static_cast<uint64_t>(mat.type());
    hash *= prime;

    for(int y = 0; y < mat.rows; ++y)
    {
        const unsigned char *row = mat.ptr<unsigned char>(y);
        const size_t bytes = static_cast<size_t>(mat.cols) * mat.elemSize();
        for(size_t i = 0; i < bytes; ++i)
        {
            hash ^= static_cast<uint64_t>(row[i]);
            hash *= prime;
        }
    }

    return hash;
}

int SemanticDebugMaxFrames()
{
    const char *value = std::getenv("LITESEMSLAM_SEMANTIC_DEBUG_MAX");
    if(!value || !*value)
        return 30;
    return std::max(0, std::atoi(value));
}

std::string HexChecksum(uint64_t checksum)
{
    std::ostringstream stream;
    stream << std::hex << std::setw(16) << std::setfill('0') << checksum;
    return stream.str();
}

void DumpSemanticDebugFrame(const std::string &debugDir,
                            unsigned long long requestId,
                            double timestamp,
                            const cv::Mat &imageBgr,
                            const cv::Mat &featureMask)
{
    if(debugDir.empty())
        return;

    const int maxFrames = SemanticDebugMaxFrames();
    if(maxFrames > 0 && requestId > static_cast<unsigned long long>(maxFrames))
        return;

    mkdir(debugDir.c_str(), 0755);

    std::ostringstream prefix;
    prefix << debugDir << "/frame_" << std::setw(6) << std::setfill('0') << requestId;

    cv::Mat dynamicMask;
    cv::compare(featureMask, 0, dynamicMask, cv::CMP_EQ);

    cv::Mat overlay = imageBgr.clone();
    if(!overlay.empty() && !dynamicMask.empty())
    {
        cv::Mat red(overlay.size(), overlay.type(), cv::Scalar(0, 0, 255));
        cv::Mat blended;
        cv::addWeighted(overlay, 0.45, red, 0.55, 0.0, blended);
        blended.copyTo(overlay, dynamicMask);
    }

    cv::imwrite(prefix.str() + "_input.jpg", imageBgr);
    cv::imwrite(prefix.str() + "_dynamic_mask.png", dynamicMask);
    cv::imwrite(prefix.str() + "_overlay.jpg", overlay);

    std::ofstream meta((prefix.str() + ".json").c_str());
    meta << "{\n";
    meta << "  \"request_id\": " << requestId << ",\n";
    meta << "  \"timestamp\": " << std::fixed << std::setprecision(6) << timestamp << ",\n";
    meta << "  \"image_checksum\": \"" << HexChecksum(MatChecksum(imageBgr)) << "\",\n";
    meta << "  \"feature_mask_checksum\": \"" << HexChecksum(MatChecksum(featureMask)) << "\",\n";
    meta << "  \"dynamic_pixels\": " << cv::countNonZero(dynamicMask) << "\n";
    meta << "}\n";
}

class ISemanticBackend
{
public:
    virtual ~ISemanticBackend() {}
    virtual void PredictFeatureMask(const cv::Mat &imageBgr, cv::Mat &featureMask) = 0;
};

#ifdef LITESEMSLAM_USE_TENSORRT_SEMANTICS
class TrtLogger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char *message) noexcept override
    {
        if(severity <= Severity::kWARNING)
            std::cerr << "[TensorRT] " << message << std::endl;
    }
};

template<typename T>
struct TrtDestroy
{
    void operator()(T *object) const
    {
        if(object)
            object->destroy();
    }
};

std::vector<char> ReadBinaryFile(const std::string &path)
{
    std::ifstream file(path.c_str(), std::ios::binary);
    if(!file)
        throw std::runtime_error("could not open TensorRT engine: " + path);
    file.seekg(0, std::ios::end);
    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> data(static_cast<size_t>(size));
    if(size > 0 && !file.read(data.data(), size))
        throw std::runtime_error("could not read TensorRT engine: " + path);
    return data;
}

void CheckCuda(cudaError_t code, const std::string &operation)
{
    if(code != cudaSuccess)
        throw std::runtime_error(operation + " failed: " + cudaGetErrorString(code));
}

size_t DataTypeSize(nvinfer1::DataType dtype)
{
    switch(dtype)
    {
        case nvinfer1::DataType::kFLOAT:
            return sizeof(float);
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kINT8:
            return 1;
        case nvinfer1::DataType::kINT32:
            return sizeof(int32_t);
        case nvinfer1::DataType::kBOOL:
            return 1;
        default:
            throw std::runtime_error("unsupported TensorRT dtype");
    }
}

uint16_t FloatToHalfBits(float value)
{
    const __half halfValue = __float2half(value);
    return __half_as_ushort(halfValue);
}

float HalfBitsToFloat(uint16_t bits)
{
    const __half halfValue = __ushort_as_half(bits);
    return __half2float(halfValue);
}

std::vector<int> DimsToVector(const nvinfer1::Dims &dims)
{
    std::vector<int> values;
    values.reserve(dims.nbDims);
    for(int i = 0; i < dims.nbDims; ++i)
        values.push_back(dims.d[i]);
    return values;
}

int64_t Volume(const std::vector<int> &dims)
{
    if(dims.empty())
        return 0;

    int64_t total = 1;
    for(int dim : dims)
    {
        if(dim < 0)
            return -1;
        total *= dim;
    }
    return total;
}

bool HasDynamicDim(const nvinfer1::Dims &dims)
{
    for(int i = 0; i < dims.nbDims; ++i)
    {
        if(dims.d[i] < 0)
            return true;
    }
    return false;
}

struct OutputTensor
{
    std::string name;
    std::vector<int> dims;
    nvinfer1::DataType dtype = nvinfer1::DataType::kFLOAT;
    void *devicePtr = NULL;
    std::vector<float> f32;
    std::vector<int32_t> i32;

    int64_t Elements() const
    {
        return Volume(dims);
    }
};

int SemanticCudaTensorType(nvinfer1::DataType dtype)
{
    if(dtype == nvinfer1::DataType::kFLOAT)
        return cuda_semantic::kTensorFloat32;
    if(dtype == nvinfer1::DataType::kHALF)
        return cuda_semantic::kTensorFloat16;
    throw std::runtime_error("unsupported TensorRT dtype for CUDA semantic decode");
}

void EnsureDeviceBuffer(void *&ptr, size_t &capacityBytes, size_t requiredBytes, const std::string &name)
{
    if(requiredBytes == 0)
        requiredBytes = 1;
    if(capacityBytes >= requiredBytes)
        return;
    if(ptr)
        cudaFree(ptr);
    ptr = NULL;
    CheckCuda(cudaMalloc(&ptr, requiredBytes), "cudaMalloc " + name);
    capacityBytes = requiredBytes;
}

struct CudaDeviceBuffer
{
    ~CudaDeviceBuffer()
    {
        if(ptr)
            cudaFree(ptr);
    }

    void Ensure(size_t requiredBytes, const std::string &name)
    {
        EnsureDeviceBuffer(ptr, bytes, requiredBytes, name);
    }

    template<typename T>
    T* As()
    {
        return static_cast<T*>(ptr);
    }

    void *ptr = NULL;
    size_t bytes = 0;
};

class DynamicOutputAllocator : public nvinfer1::IOutputAllocator
{
public:
    ~DynamicOutputAllocator() override
    {
        Clear();
    }

    void Clear()
    {
        for(std::map<std::string, void*>::iterator it = mPtrs.begin(); it != mPtrs.end(); ++it)
        {
            if(it->second)
                cudaFree(it->second);
        }
        mPtrs.clear();
        mSizes.clear();
        mShapes.clear();
    }

    void ResetShapes()
    {
        mShapes.clear();
    }

    void* reallocateOutput(char const* tensorName, void* currentMemory, uint64_t size, uint64_t) noexcept override
    {
        const std::string name(tensorName);
        std::map<std::string, size_t>::const_iterator sizeIt = mSizes.find(name);
        if(currentMemory && sizeIt != mSizes.end() && sizeIt->second >= size)
        {
            mPtrs[name] = currentMemory;
            return currentMemory;
        }

        std::map<std::string, void*>::iterator ptrIt = mPtrs.find(name);
        if(ptrIt != mPtrs.end() && ptrIt->second)
        {
            cudaFree(ptrIt->second);
            ptrIt->second = NULL;
        }

        void *ptr = NULL;
        if(cudaMalloc(&ptr, static_cast<size_t>(size)) != cudaSuccess)
            return NULL;

        mPtrs[name] = ptr;
        mSizes[name] = static_cast<size_t>(size);
        return ptr;
    }

    void notifyShape(char const* tensorName, nvinfer1::Dims const& dims) noexcept override
    {
        mShapes[std::string(tensorName)] = dims;
    }

    void* Ptr(const std::string &name) const
    {
        std::map<std::string, void*>::const_iterator it = mPtrs.find(name);
        return it == mPtrs.end() ? NULL : it->second;
    }

    bool Shape(const std::string &name, nvinfer1::Dims &dims) const
    {
        std::map<std::string, nvinfer1::Dims>::const_iterator it = mShapes.find(name);
        if(it == mShapes.end())
            return false;
        dims = it->second;
        return true;
    }

private:
    std::map<std::string, void*> mPtrs;
    std::map<std::string, size_t> mSizes;
    std::map<std::string, nvinfer1::Dims> mShapes;
};

class TensorRTEngine
{
public:
    TensorRTEngine(const std::string &enginePath, const std::string &pluginPath, int deviceId)
    {
        CheckCuda(cudaSetDevice(deviceId), "cudaSetDevice");

        if(!pluginPath.empty())
        {
            mPluginHandle = dlopen(pluginPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
            if(!mPluginHandle)
                throw std::runtime_error("could not load TensorRT plugin " + pluginPath + ": " + dlerror());
        }

        initLibNvInferPlugins(&mLogger, "");
        std::vector<char> engineData = ReadBinaryFile(enginePath);

        mRuntime.reset(nvinfer1::createInferRuntime(mLogger));
        if(!mRuntime)
            throw std::runtime_error("could not create TensorRT runtime");

        mEngine.reset(mRuntime->deserializeCudaEngine(engineData.data(), engineData.size()));
        if(!mEngine)
            throw std::runtime_error("could not deserialize TensorRT engine: " + enginePath);

        mContext.reset(mEngine->createExecutionContext());
        if(!mContext)
            throw std::runtime_error("could not create TensorRT execution context");

        int leastPriority = 0;
        int greatestPriority = 0;
        CheckCuda(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority),
                  "cudaDeviceGetStreamPriorityRange");
        CheckCuda(cudaStreamCreateWithPriority(&mStream, cudaStreamNonBlocking, greatestPriority),
                  "cudaStreamCreateWithPriority semantic");
        CheckCuda(cudaEventCreateWithFlags(&mInferStart, cudaEventDefault), "cudaEventCreate inference start");
        CheckCuda(cudaEventCreateWithFlags(&mInferEnd, cudaEventDefault), "cudaEventCreate inference end");

        for(int i = 0; i < mEngine->getNbIOTensors(); ++i)
        {
            const char *tensorName = mEngine->getIOTensorName(i);
            if(!tensorName)
                continue;
            const std::string name(tensorName);
            if(mEngine->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kINPUT)
                mInputNames.push_back(name);
            else
                mOutputNames.push_back(name);
        }

        if(mInputNames.empty() || mOutputNames.empty())
            throw std::runtime_error("TensorRT engine has no usable inputs or outputs");
    }

    ~TensorRTEngine()
    {
        for(std::map<std::string, void*>::iterator it = mFixedPtrs.begin(); it != mFixedPtrs.end(); ++it)
        {
            if(it->second)
                cudaFree(it->second);
        }
        if(mInferStart)
            cudaEventDestroy(mInferStart);
        if(mInferEnd)
            cudaEventDestroy(mInferEnd);
        if(mStream)
            cudaStreamDestroy(mStream);
        if(mPluginHandle)
            dlclose(mPluginHandle);
    }

    const std::string& InputName() const
    {
        return mInputNames.front();
    }

    cudaStream_t Stream() const
    {
        return mStream;
    }

    float LastInferenceGpuMs() const
    {
        if(!mInferStart || !mInferEnd)
            return 0.0f;
        float ms = 0.0f;
        if(cudaEventElapsedTime(&ms, mInferStart, mInferEnd) != cudaSuccess)
            return 0.0f;
        return ms;
    }

    nvinfer1::DataType InputDataType() const
    {
        return mEngine->getTensorDataType(InputName().c_str());
    }

    void* PrepareInputDeviceBuffer(const nvinfer1::Dims &inputDims)
    {
        const std::string inputName = InputName();
        if(!mContext->setInputShape(inputName.c_str(), inputDims))
            throw std::runtime_error("TensorRT setInputShape failed for " + inputName);

        const nvinfer1::DataType inputDtype = mEngine->getTensorDataType(inputName.c_str());
        if(inputDtype != nvinfer1::DataType::kFLOAT && inputDtype != nvinfer1::DataType::kHALF)
            throw std::runtime_error("unsupported TensorRT input dtype for " + inputName);

        const int64_t inputElements = Volume(DimsToVector(inputDims));
        if(inputElements <= 0)
            throw std::runtime_error("invalid TensorRT input shape for " + inputName);
        const size_t inputBytes = static_cast<size_t>(inputElements) * DataTypeSize(inputDtype);
        return EnsureFixedBuffer(inputName, inputBytes);
    }

    std::vector<OutputTensor> InferDeviceFromDeviceInput(void *inputPtr, const nvinfer1::Dims &inputDims)
    {
        const std::string inputName = InputName();
        if(!inputPtr)
            throw std::runtime_error("TensorRT input device buffer is null");
        if(!mContext->setInputShape(inputName.c_str(), inputDims))
            throw std::runtime_error("TensorRT setInputShape failed for " + inputName);
        if(!mContext->setTensorAddress(inputName.c_str(), inputPtr))
            throw std::runtime_error("TensorRT setTensorAddress failed for " + inputName);

        mAllocator.ResetShapes();
        for(size_t i = 0; i < mOutputNames.size(); ++i)
        {
            const std::string &name = mOutputNames[i];
            nvinfer1::Dims engineDims = mEngine->getTensorShape(name.c_str());
            if(HasDynamicDim(engineDims))
            {
                if(!mContext->setOutputAllocator(name.c_str(), &mAllocator))
                    throw std::runtime_error("TensorRT setOutputAllocator failed for " + name);
            }
            else
            {
                nvinfer1::Dims dims = mContext->getTensorShape(name.c_str());
                const nvinfer1::DataType dtype = mEngine->getTensorDataType(name.c_str());
                const int64_t elements = Volume(DimsToVector(dims));
                void *ptr = EnsureFixedBuffer(name, static_cast<size_t>(elements) * DataTypeSize(dtype));
                if(!mContext->setTensorAddress(name.c_str(), ptr))
                    throw std::runtime_error("TensorRT setTensorAddress failed for " + name);
            }
        }

        CheckCuda(cudaEventRecord(mInferStart, mStream), "cudaEventRecord semantic inference start");
        if(!mContext->enqueueV3(mStream))
            throw std::runtime_error("TensorRT enqueueV3 failed");
        CheckCuda(cudaEventRecord(mInferEnd, mStream), "cudaEventRecord semantic inference end");

        std::vector<OutputTensor> outputs;
        outputs.reserve(mOutputNames.size());
        for(size_t i = 0; i < mOutputNames.size(); ++i)
        {
            const std::string &name = mOutputNames[i];
            nvinfer1::Dims dims;
            void *devicePtr = NULL;
            if(mAllocator.Shape(name, dims))
                devicePtr = mAllocator.Ptr(name);
            else
            {
                dims = mContext->getTensorShape(name.c_str());
                devicePtr = mFixedPtrs[name];
            }

            OutputTensor tensor;
            tensor.name = name;
            tensor.dims = DimsToVector(dims);
            tensor.dtype = mEngine->getTensorDataType(name.c_str());
            tensor.devicePtr = devicePtr;

            const int64_t elements = tensor.Elements();
            if(elements <= 0 || !devicePtr)
            {
                outputs.push_back(tensor);
                continue;
            }
            outputs.push_back(tensor);
        }

        return outputs;
    }

    std::vector<OutputTensor> InferDevice(const std::vector<float> &input, const nvinfer1::Dims &inputDims)
    {
        const std::string inputName = InputName();
        const nvinfer1::DataType inputDtype = mEngine->getTensorDataType(inputName.c_str());
        size_t inputBytes = 0;
        const void *inputHost = NULL;
        if(inputDtype == nvinfer1::DataType::kFLOAT)
        {
            inputBytes = input.size() * sizeof(float);
            inputHost = input.data();
        }
        else if(inputDtype == nvinfer1::DataType::kHALF)
        {
            mInputHalfHost.resize(input.size());
            for(size_t i = 0; i < input.size(); ++i)
                mInputHalfHost[i] = FloatToHalfBits(input[i]);
            inputBytes = mInputHalfHost.size() * sizeof(uint16_t);
            inputHost = mInputHalfHost.data();
        }
        else
        {
            throw std::runtime_error("unsupported TensorRT input dtype for " + inputName);
        }

        void *inputPtr = PrepareInputDeviceBuffer(inputDims);
        CheckCuda(cudaMemcpyAsync(inputPtr, inputHost, inputBytes, cudaMemcpyHostToDevice, mStream),
                  "cudaMemcpyAsync H2D");
        return InferDeviceFromDeviceInput(inputPtr, inputDims);
    }

    std::vector<OutputTensor> Infer(const std::vector<float> &input, const nvinfer1::Dims &inputDims)
    {
        std::vector<OutputTensor> outputs = InferDevice(input, inputDims);
        for(size_t i = 0; i < outputs.size(); ++i)
        {
            OutputTensor &tensor = outputs[i];
            const int64_t elements = tensor.Elements();
            if(elements <= 0 || !tensor.devicePtr)
                continue;
            if(tensor.dtype == nvinfer1::DataType::kFLOAT)
            {
                tensor.f32.resize(static_cast<size_t>(elements));
                CheckCuda(cudaMemcpyAsync(tensor.f32.data(), tensor.devicePtr,
                                          tensor.f32.size() * sizeof(float),
                                          cudaMemcpyDeviceToHost, mStream),
                          "cudaMemcpyAsync D2H float");
            }
            else if(tensor.dtype == nvinfer1::DataType::kHALF)
            {
                std::vector<uint16_t> halfValues(static_cast<size_t>(elements));
                CheckCuda(cudaMemcpyAsync(halfValues.data(), tensor.devicePtr,
                                          halfValues.size() * sizeof(uint16_t),
                                          cudaMemcpyDeviceToHost, mStream),
                          "cudaMemcpyAsync D2H half");
                CheckCuda(cudaStreamSynchronize(mStream), "cudaStreamSynchronize half output");

                tensor.f32.resize(static_cast<size_t>(elements));
                for(size_t j = 0; j < halfValues.size(); ++j)
                    tensor.f32[j] = HalfBitsToFloat(halfValues[j]);
            }
            else if(tensor.dtype == nvinfer1::DataType::kINT32)
            {
                tensor.i32.resize(static_cast<size_t>(elements));
                CheckCuda(cudaMemcpyAsync(tensor.i32.data(), tensor.devicePtr,
                                          tensor.i32.size() * sizeof(int32_t),
                                          cudaMemcpyDeviceToHost, mStream),
                          "cudaMemcpyAsync D2H int32");
            }
            else
            {
                throw std::runtime_error("unsupported TensorRT output dtype for " + tensor.name);
            }
        }

        CheckCuda(cudaStreamSynchronize(mStream), "cudaStreamSynchronize outputs");
        return outputs;
    }

private:
    void* EnsureFixedBuffer(const std::string &name, size_t bytes)
    {
        if(bytes == 0)
            bytes = 1;

        if(mFixedSizes[name] < bytes)
        {
            if(mFixedPtrs[name])
                cudaFree(mFixedPtrs[name]);
            void *ptr = NULL;
        CheckCuda(cudaMalloc(&ptr, bytes), "cudaMalloc");
        mFixedPtrs[name] = ptr;
        mFixedSizes[name] = bytes;
        }
        return mFixedPtrs[name];
    }

    TrtLogger mLogger;
    void *mPluginHandle = NULL;
    cudaStream_t mStream = NULL;
    cudaEvent_t mInferStart = NULL;
    cudaEvent_t mInferEnd = NULL;
    std::unique_ptr<nvinfer1::IRuntime, TrtDestroy<nvinfer1::IRuntime> > mRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine, TrtDestroy<nvinfer1::ICudaEngine> > mEngine;
    std::unique_ptr<nvinfer1::IExecutionContext, TrtDestroy<nvinfer1::IExecutionContext> > mContext;
    DynamicOutputAllocator mAllocator;
    std::vector<std::string> mInputNames;
    std::vector<std::string> mOutputNames;
    std::map<std::string, void*> mFixedPtrs;
    std::map<std::string, size_t> mFixedSizes;
    std::vector<uint16_t> mInputHalfHost;
};

class SegNetTensorRTBackend : public ISemanticBackend
{
public:
    explicit SegNetTensorRTBackend(const SemanticMaskerConfig &config)
        : mConfig(config),
          mEngine(config.segnetEnginePath.empty()
                      ? JoinPath(config.segmentationRoot, "models/final/segnet_exact.engine")
                      : config.segnetEnginePath,
                  config.segnetPluginPath.empty()
                      ? JoinPath(config.segmentationRoot, "plugins/segnet_unpool/libsegnet_trt_plugins.so")
                      : config.segnetPluginPath,
                  config.deviceId)
    {
        std::map<std::string, int> pascal;
        pascal["aeroplane"] = 1;
        pascal["bicycle"] = 2;
        pascal["bird"] = 3;
        pascal["boat"] = 4;
        pascal["bus"] = 6;
        pascal["car"] = 7;
        pascal["cat"] = 8;
        pascal["cow"] = 10;
        pascal["dog"] = 12;
        pascal["horse"] = 13;
        pascal["motorbike"] = 14;
        pascal["motorcycle"] = 14;
        pascal["person"] = 15;
        pascal["sheep"] = 17;
        pascal["train"] = 19;

        const std::set<int> defaults = {2, 3, 6, 7, 8, 10, 12, 13, 14, 15, 17, 19};
        mDynamicLabels = ParseDynamicLabels(config.dynamicLabels, pascal, defaults);
    }

    void PredictFeatureMask(const cv::Mat &imageBgr, cv::Mat &featureMask) override
    {
        cv::Mat resized;
        cv::resize(imageBgr, resized, cv::Size(224, 224), 0.0, 0.0, cv::INTER_LINEAR);

        std::vector<float> input(3 * 224 * 224);
        for(int y = 0; y < 224; ++y)
        {
            const cv::Vec3b *row = resized.ptr<cv::Vec3b>(y);
            for(int x = 0; x < 224; ++x)
            {
                const int index = y * 224 + x;
                input[index] = static_cast<float>(row[x][0]);
                input[224 * 224 + index] = static_cast<float>(row[x][1]);
                input[2 * 224 * 224 + index] = static_cast<float>(row[x][2]);
            }
        }

        std::vector<OutputTensor> outputs = mEngine.Infer(input, nvinfer1::Dims4(1, 3, 224, 224));
        const OutputTensor *logits = NULL;
        for(size_t i = 0; i < outputs.size(); ++i)
        {
            if(outputs[i].dtype == nvinfer1::DataType::kFLOAT && outputs[i].Elements() == 21 * 224 * 224)
            {
                logits = &outputs[i];
                break;
            }
        }
        if(!logits)
            throw std::runtime_error("SegNet TensorRT output tensor not found");

        cv::Mat dynamicSmall(224, 224, CV_8UC1, cv::Scalar(0));
        const float *values = logits->f32.data();
        const int hw = 224 * 224;
        for(int index = 0; index < hw; ++index)
        {
            int bestClass = 0;
            float bestValue = values[index];
            for(int c = 1; c < 21; ++c)
            {
                const float v = values[c * hw + index];
                if(v > bestValue)
                {
                    bestValue = v;
                    bestClass = c;
                }
            }

            if(mDynamicLabels.find(bestClass) == mDynamicLabels.end())
                continue;

            float sum = 0.0f;
            for(int c = 0; c < 21; ++c)
                sum += std::exp(values[c * hw + index] - bestValue);
            const float probability = sum > 0.0f ? 1.0f / sum : 0.0f;
            if(probability >= mConfig.segnetThreshold)
                dynamicSmall.ptr<unsigned char>(index / 224)[index % 224] = 255;
        }

        cv::Mat dynamicFull;
        cv::resize(dynamicSmall, dynamicFull, imageBgr.size(), 0.0, 0.0, cv::INTER_NEAREST);
        DynamicToFeatureMask(dynamicFull, mConfig.dilatePixels, featureMask);
    }

private:
    SemanticMaskerConfig mConfig;
    TensorRTEngine mEngine;
    std::set<int> mDynamicLabels;
};

class MaskRCNNTensorRTBackend : public ISemanticBackend
{
public:
    explicit MaskRCNNTensorRTBackend(const SemanticMaskerConfig &config)
        : mConfig(config),
          mEngine(config.maskrcnnEnginePath.empty()
                      ? JoinPath(config.segmentationRoot, "models/final/maskrcnn_fp16.engine")
                      : config.maskrcnnEnginePath,
                  std::string(),
                  config.deviceId)
    {
        std::map<std::string, int> coco;
        coco["person"] = 1;
        coco["bicycle"] = 2;
        coco["car"] = 3;
        coco["motorcycle"] = 4;
        coco["motorbike"] = 4;
        coco["bus"] = 6;
        coco["train"] = 7;
        coco["truck"] = 8;
        coco["bird"] = 15;
        coco["cat"] = 16;
        coco["dog"] = 17;
        coco["horse"] = 18;
        coco["sheep"] = 19;
        coco["cow"] = 20;

        const std::set<int> defaults = {1, 2, 3, 4, 6, 7, 8, 15, 16, 17, 18, 19, 20};
        mDynamicLabels = ParseDynamicLabels(config.dynamicLabels, coco, defaults);
    }

    void PredictFeatureMask(const cv::Mat &imageBgr, cv::Mat &featureMask) override
    {
        const int height = imageBgr.rows;
        const int width = imageBgr.cols;
        const float ratio = static_cast<float>(mConfig.maskrcnnMinSide) / static_cast<float>(std::min(width, height));
        const int resizedWidth = static_cast<int>(ratio * width);
        const int resizedHeight = static_cast<int>(ratio * height);

        cv::Mat resized;
        cv::resize(imageBgr, resized, cv::Size(resizedWidth, resizedHeight), 0.0, 0.0, cv::INTER_LINEAR);

        const int paddedHeight = ((resizedHeight + 31) / 32) * 32;
        const int paddedWidth = ((resizedWidth + 31) / 32) * 32;
        std::vector<float> input(3 * paddedHeight * paddedWidth, 0.0f);
        const float mean[3] = {102.9801f, 115.9465f, 122.7717f};
        for(int y = 0; y < resizedHeight; ++y)
        {
            const cv::Vec3b *row = resized.ptr<cv::Vec3b>(y);
            for(int x = 0; x < resizedWidth; ++x)
            {
                const int index = y * paddedWidth + x;
                input[index] = static_cast<float>(row[x][0]) - mean[0];
                input[paddedHeight * paddedWidth + index] = static_cast<float>(row[x][1]) - mean[1];
                input[2 * paddedHeight * paddedWidth + index] = static_cast<float>(row[x][2]) - mean[2];
            }
        }

        std::vector<OutputTensor> outputs = mEngine.Infer(input, nvinfer1::Dims3(3, paddedHeight, paddedWidth));
        const OutputTensor *boxes = NULL;
        const OutputTensor *labels = NULL;
        const OutputTensor *scores = NULL;
        const OutputTensor *masks = NULL;

        for(size_t i = 0; i < outputs.size(); ++i)
        {
            const OutputTensor &tensor = outputs[i];
            if(tensor.dtype == nvinfer1::DataType::kINT32 && tensor.dims.size() == 1)
                labels = &tensor;
            else if(tensor.dtype == nvinfer1::DataType::kFLOAT && tensor.dims.size() == 2 && tensor.dims[1] == 4)
                boxes = &tensor;
            else if(tensor.dtype == nvinfer1::DataType::kFLOAT && tensor.dims.size() == 1)
                scores = &tensor;
            else if(tensor.dtype == nvinfer1::DataType::kFLOAT && tensor.dims.size() == 4 &&
                    tensor.dims[1] == 1 && tensor.dims[2] == 28 && tensor.dims[3] == 28)
                masks = &tensor;
        }

        if(!boxes || !labels || !scores || !masks)
            throw std::runtime_error("Mask R-CNN TensorRT output tensors not found");

        const int detections = std::min<int>(labels->i32.size(), scores->f32.size());
        cv::Mat dynamic(height, width, CV_8UC1, cv::Scalar(0));
        for(int i = 0; i < detections; ++i)
        {
            if(mDynamicLabels.find(labels->i32[i]) == mDynamicLabels.end())
                continue;
            if(scores->f32[i] < mConfig.maskrcnnThreshold)
                continue;

            const float *box = boxes->f32.data() + 4 * i;
            int x0 = static_cast<int>(std::round(box[0] / ratio));
            int y0 = static_cast<int>(std::round(box[1] / ratio));
            int x1 = static_cast<int>(std::round(box[2] / ratio));
            int y1 = static_cast<int>(std::round(box[3] / ratio));
            x0 = std::max(0, std::min(width - 1, x0));
            y0 = std::max(0, std::min(height - 1, y0));
            x1 = std::max(0, std::min(width - 1, x1));
            y1 = std::max(0, std::min(height - 1, y1));
            if(x1 <= x0 || y1 <= y0)
                continue;

            const float *maskData = masks->f32.data() + i * 28 * 28;
            cv::Mat mask28(28, 28, CV_32FC1, const_cast<float*>(maskData));
            cv::Mat resizedMask;
            cv::resize(mask28, resizedMask, cv::Size(x1 - x0 + 1, y1 - y0 + 1), 0.0, 0.0, cv::INTER_LINEAR);

            cv::Mat roi = dynamic(cv::Rect(x0, y0, x1 - x0 + 1, y1 - y0 + 1));
            roi.setTo(cv::Scalar(255), resizedMask > 0.5f);
        }

        DynamicToFeatureMask(dynamic, mConfig.dilatePixels, featureMask);
    }

private:
    SemanticMaskerConfig mConfig;
    TensorRTEngine mEngine;
    std::set<int> mDynamicLabels;
};

class TokenMaskTensorRTBackend : public ISemanticBackend
{
public:
    explicit TokenMaskTensorRTBackend(const SemanticMaskerConfig &config)
        : mConfig(config),
          mInputSize(std::max(32, config.tokenmaskInputWidth),
                     std::max(32, config.tokenmaskInputHeight)),
          mDataset(ResolveDatasetName(config)),
          mEngine(ResolveEnginePath(config), std::string(), config.deviceId)
    {
        std::map<std::string, int> names;
        std::set<int> defaults;

        if(mDataset == "city" || mDataset == "cityscapes")
        {
            mDataset = "cityscapes";
            names["person"] = 11;
            names["rider"] = 12;
            names["car"] = 13;
            names["truck"] = 14;
            names["bus"] = 15;
            names["train"] = 16;
            names["motorcycle"] = 17;
            names["motorbike"] = 17;
            names["bicycle"] = 18;
            defaults.insert(11);
            defaults.insert(12);
            defaults.insert(13);
            defaults.insert(14);
            defaults.insert(15);
            defaults.insert(16);
            defaults.insert(17);
            defaults.insert(18);
        }
        else
        {
            mDataset = "ade20k";
            names["person"] = 12;
            names["car"] = 20;
            names["boat"] = 76;
            names["bus"] = 80;
            names["truck"] = 83;
            names["airplane"] = 90;
            names["aeroplane"] = 90;
            names["van"] = 102;
            names["ship"] = 103;
            names["minibike"] = 116;
            names["motorbike"] = 116;
            names["motorcycle"] = 116;
            names["animal"] = 126;
            names["bicycle"] = 127;
            defaults.insert(12);
            defaults.insert(20);
            defaults.insert(76);
            defaults.insert(80);
            defaults.insert(83);
            defaults.insert(90);
            defaults.insert(102);
            defaults.insert(103);
            defaults.insert(116);
            defaults.insert(126);
            defaults.insert(127);
        }

        mDynamicLabels = ParseDynamicLabels(config.dynamicLabels, names, defaults);
        WarmUp();
    }

    void PredictFeatureMask(const cv::Mat &imageBgr, cv::Mat &featureMask) override
    {
        if(imageBgr.type() != CV_8UC3)
            throw std::runtime_error("TokenMask backend expects CV_8UC3 BGR input");

        const bool trace = !mWarmingUp && SemanticTraceEnabled();
        const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        const InputProjection projection = ComputeInputProjection(imageBgr.size());
        const std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        const size_t compactStepBytes = static_cast<size_t>(imageBgr.cols) * 3;
        mInputImageDevice.Ensure(compactStepBytes * static_cast<size_t>(imageBgr.rows),
                                 "tokenmask input image");
        CheckCuda(cudaMemcpy2DAsync(mInputImageDevice.As<unsigned char>(),
                                    compactStepBytes,
                                    imageBgr.data,
                                    static_cast<size_t>(imageBgr.step),
                                    compactStepBytes,
                                    static_cast<size_t>(imageBgr.rows),
                                    cudaMemcpyHostToDevice,
                                    mEngine.Stream()),
                  "cudaMemcpy2DAsync tokenmask image H2D");
        const std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

        const nvinfer1::Dims4 inputDims(1, 3, mInputSize.height, mInputSize.width);
        void *inputDevice = mEngine.PrepareInputDeviceBuffer(inputDims);
        const cudaError_t prepareStatus = cuda_semantic::PrepareTokenMaskInputFromBgr(
            mInputImageDevice.As<unsigned char>(),
            imageBgr.cols,
            imageBgr.rows,
            compactStepBytes,
            inputDevice,
            SemanticCudaTensorType(mEngine.InputDataType()),
            mInputSize.width,
            mInputSize.height,
            projection.resizedWidth,
            projection.resizedHeight,
            mEngine.Stream());
        CheckCuda(prepareStatus, "PrepareTokenMaskInputFromBgr launch");
        const std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();

        std::vector<OutputTensor> outputs =
            mEngine.InferDeviceFromDeviceInput(inputDevice, inputDims);
        const std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();

        const OutputTensor *maskLogits = NULL;
        const OutputTensor *classLogits = NULL;
        SelectOutputs(outputs, maskLogits, classLogits);
        if(!maskLogits || !classLogits)
            throw std::runtime_error("TokenMask TensorRT mask/class output tensors not found");

        DecodeFeatureMaskOnGpu(*maskLogits, *classLogits, projection, imageBgr.size(), featureMask);
        const std::chrono::steady_clock::time_point t5 = std::chrono::steady_clock::now();

        if(trace)
        {
            std::cerr << "SemanticTrace tokenmask "
                      << "projection_ms=" << MsSince(t0, t1) << " "
                      << "upload_enqueue_ms=" << MsSince(t1, t2) << " "
                      << "preprocess_enqueue_ms=" << MsSince(t2, t3) << " "
                      << "infer_enqueue_ms=" << MsSince(t3, t4) << " "
                      << "gpu_sync_decode_ms=" << MsSince(t4, t5) << " "
                      << "trt_gpu_ms=" << mEngine.LastInferenceGpuMs() << std::endl;
        }
    }

private:
    void WarmUp()
    {
        cv::Mat warmImage(mInputSize.height, mInputSize.width, CV_8UC3, cv::Scalar::all(0));
        cv::Mat warmMask;
        mWarmingUp = true;
        try
        {
            PredictFeatureMask(warmImage, warmMask);
        }
        catch(...)
        {
            mWarmingUp = false;
            throw;
        }
        mWarmingUp = false;
    }

    struct InputProjection
    {
        double scale = 1.0;
        int resizedWidth = 0;
        int resizedHeight = 0;
    };

    static std::string ResolveDatasetName(const SemanticMaskerConfig &config)
    {
        std::string dataset = ToLower(config.tokenmaskDataset);
        if(dataset.empty() || dataset == "auto")
        {
            const std::string path = ToLower(config.tokenmaskEnginePath.empty()
                                                ? config.tokenmaskModelPath
                                                : config.tokenmaskEnginePath);
            dataset = ContainsToken(path, "_city") || ContainsToken(path, "cityscapes")
                          ? "cityscapes"
                          : "ade20k";
        }
        return dataset;
    }

    static std::string ResolveEnginePath(const SemanticMaskerConfig &config)
    {
        if(!config.tokenmaskEnginePath.empty())
            return config.tokenmaskEnginePath;
        if(!config.tokenmaskModelPath.empty() && EndsWithToken(config.tokenmaskModelPath, ".engine"))
            return config.tokenmaskModelPath;
        if(!config.tokenmaskModelPath.empty() && EndsWithToken(config.tokenmaskModelPath, ".onnx"))
            throw std::runtime_error("TokenMask backend requires Semantic.tokenmask_engine for TensorRT; "
                                     "build it with tools/build_tokenmask_tensorrt_engines.sh");
        return "/home/nvidia/Desktop/kalvici/trt_cache/ours_tiny_640x640_fp16.engine";
    }

    InputProjection ComputeInputProjection(const cv::Size &imageSize) const
    {
        InputProjection projection;
        projection.scale = std::min(
            mInputSize.height / static_cast<double>(imageSize.height),
            mInputSize.width / static_cast<double>(imageSize.width));
        projection.resizedWidth = std::max(1, static_cast<int>(std::round(imageSize.width * projection.scale)));
        projection.resizedHeight = std::max(1, static_cast<int>(std::round(imageSize.height * projection.scale)));
        projection.resizedWidth = std::min(projection.resizedWidth, mInputSize.width);
        projection.resizedHeight = std::min(projection.resizedHeight, mInputSize.height);
        return projection;
    }

    cv::Mat PrepareInputImage(const cv::Mat &imageBgr, InputProjection &projection) const
    {
        projection = ComputeInputProjection(imageBgr.size());

        cv::Mat resized;
        cv::resize(imageBgr, resized, cv::Size(projection.resizedWidth, projection.resizedHeight),
                   0.0, 0.0, cv::INTER_LINEAR);

        cv::Mat padded(mInputSize, CV_8UC3, cv::Scalar::all(0));
        resized.copyTo(padded(cv::Rect(0, 0, resized.cols, resized.rows)));
        return padded;
    }

    std::vector<float> PrepareInputTensor(const cv::Mat &imageBgr) const
    {
        const int width = imageBgr.cols;
        const int height = imageBgr.rows;
        const size_t plane = static_cast<size_t>(width) * static_cast<size_t>(height);

        std::vector<float> input(plane * 3);
        for(int y = 0; y < height; ++y)
        {
            const cv::Vec3b *row = imageBgr.ptr<cv::Vec3b>(y);
            for(int x = 0; x < width; ++x)
            {
                const size_t index = static_cast<size_t>(y) * static_cast<size_t>(width) +
                                     static_cast<size_t>(x);
                input[index] = static_cast<float>(row[x][2]) / 255.0f;
                input[plane + index] = static_cast<float>(row[x][1]) / 255.0f;
                input[2 * plane + index] = static_cast<float>(row[x][0]) / 255.0f;
            }
        }
        return input;
    }

    void SelectOutputs(const std::vector<OutputTensor> &outputs,
                       const OutputTensor *&maskLogits,
                       const OutputTensor *&classLogits) const
    {
        for(size_t i = 0; i < outputs.size(); ++i)
        {
            const OutputTensor &tensor = outputs[i];
            if(tensor.Elements() <= 0 || !tensor.devicePtr)
                continue;
            if(ContainsToken(tensor.name, "mask"))
                maskLogits = &tensor;
            else if(ContainsToken(tensor.name, "class"))
                classLogits = &tensor;
        }

        if(!maskLogits)
        {
            int64_t bestVolume = -1;
            for(size_t i = 0; i < outputs.size(); ++i)
            {
                const OutputTensor &tensor = outputs[i];
                if(tensor.Elements() <= 0 || !tensor.devicePtr || tensor.dims.size() < 4)
                    continue;
                const int64_t volume = tensor.Elements();
                if(volume > bestVolume)
                {
                    bestVolume = volume;
                    maskLogits = &tensor;
                }
            }
        }

        if(!classLogits)
        {
            int64_t bestVolume = std::numeric_limits<int64_t>::max();
            for(size_t i = 0; i < outputs.size(); ++i)
            {
                const OutputTensor &tensor = outputs[i];
                if(tensor.Elements() <= 0 || !tensor.devicePtr || &tensor == maskLogits || tensor.dims.size() < 2)
                    continue;
                const int64_t volume = tensor.Elements();
                if(volume < bestVolume)
                {
                    bestVolume = volume;
                    classLogits = &tensor;
                }
            }
        }
    }

    void DecodeFeatureMaskOnGpu(const OutputTensor &maskLogits,
                                const OutputTensor &classLogits,
                                const InputProjection &projection,
                                const cv::Size &imageSize,
                                cv::Mat &featureMask)
    {
        if(maskLogits.dims.size() < 4 || classLogits.dims.size() < 3)
            throw std::runtime_error("Unexpected TokenMask output rank");

        const int queryCount = maskLogits.dims[maskLogits.dims.size() - 3];
        const int maskHeight = maskLogits.dims[maskLogits.dims.size() - 2];
        const int maskWidth = maskLogits.dims[maskLogits.dims.size() - 1];
        const int classCount = classLogits.dims.back();
        const int imageWidth = imageSize.width;
        const int imageHeight = imageSize.height;
        const int area = maskHeight * maskWidth;

        if(queryCount <= 0 || maskHeight <= 0 || maskWidth <= 0 || classCount <= 1 ||
           imageWidth <= 0 || imageHeight <= 0)
            throw std::runtime_error("Invalid TokenMask output shapes");
        if(maskLogits.Elements() < static_cast<int64_t>(queryCount) *
                                   static_cast<int64_t>(maskHeight) *
                                   static_cast<int64_t>(maskWidth) ||
           classLogits.Elements() < static_cast<int64_t>(queryCount) *
                                    static_cast<int64_t>(classCount))
            throw std::runtime_error("TokenMask output tensors are smaller than their decoded shapes");

        EnsureDynamicClassMask(classCount);
        mQueryClass.Ensure(sizeof(int) * static_cast<size_t>(queryCount), "tokenmask query class");
        mQueryScore.Ensure(sizeof(float) * static_cast<size_t>(queryCount), "tokenmask query score");
        mQueryKeep.Ensure(sizeof(unsigned char) * static_cast<size_t>(queryCount), "tokenmask query keep");
        mBestQuery.Ensure(sizeof(int) * static_cast<size_t>(area), "tokenmask best query");
        mOriginalArea.Ensure(sizeof(int) * static_cast<size_t>(queryCount), "tokenmask original area");
        mWinnerArea.Ensure(sizeof(int) * static_cast<size_t>(queryCount), "tokenmask winner area");
        mFinalArea.Ensure(sizeof(int) * static_cast<size_t>(queryCount), "tokenmask final area");
        mQueryAccepted.Ensure(sizeof(unsigned char) * static_cast<size_t>(queryCount), "tokenmask accepted query");
        mDynamicSmall.Ensure(sizeof(unsigned char) * static_cast<size_t>(area), "tokenmask dynamic small");
        mFeatureMaskDevice.Ensure(sizeof(unsigned char) *
                                  static_cast<size_t>(imageWidth) * static_cast<size_t>(imageHeight),
                                  "tokenmask feature mask");

        const cudaError_t launchStatus = cuda_semantic::DecodeTokenMaskFeatureMask(
            maskLogits.devicePtr,
            SemanticCudaTensorType(maskLogits.dtype),
            classLogits.devicePtr,
            SemanticCudaTensorType(classLogits.dtype),
            queryCount,
            maskHeight,
            maskWidth,
            classCount,
            mDynamicClassMaskDevice.As<unsigned char>(),
            mConfig.tokenmaskMaskThreshold,
            mConfig.tokenmaskOverlapThreshold,
            mInputSize.width,
            mInputSize.height,
            std::max(1, std::min(projection.resizedWidth, mInputSize.width)),
            std::max(1, std::min(projection.resizedHeight, mInputSize.height)),
            imageWidth,
            imageHeight,
            mConfig.dilatePixels,
            mQueryClass.As<int>(),
            mQueryScore.As<float>(),
            mQueryKeep.As<unsigned char>(),
            mBestQuery.As<int>(),
            mOriginalArea.As<int>(),
            mWinnerArea.As<int>(),
            mFinalArea.As<int>(),
            mQueryAccepted.As<unsigned char>(),
            mDynamicSmall.As<unsigned char>(),
            mFeatureMaskDevice.As<unsigned char>(),
            mEngine.Stream());
        CheckCuda(launchStatus, "DecodeTokenMaskFeatureMask launch");

        featureMask.create(imageHeight, imageWidth, CV_8UC1);
        CheckCuda(cudaMemcpyAsync(featureMask.data,
                                  mFeatureMaskDevice.As<unsigned char>(),
                                  static_cast<size_t>(imageWidth) * static_cast<size_t>(imageHeight),
                                  cudaMemcpyDeviceToHost,
                                  mEngine.Stream()),
                  "cudaMemcpyAsync tokenmask feature mask D2H");
        CheckCuda(cudaStreamSynchronize(mEngine.Stream()), "cudaStreamSynchronize tokenmask decode");
    }

    void EnsureDynamicClassMask(int classCount)
    {
        if(classCount == mDynamicClassMaskClasses)
            return;

        std::vector<unsigned char> host(static_cast<size_t>(classCount), 0);
        for(std::set<int>::const_iterator it = mDynamicLabels.begin(); it != mDynamicLabels.end(); ++it)
        {
            if(*it >= 0 && *it < classCount)
                host[static_cast<size_t>(*it)] = 1;
        }

        mDynamicClassMaskDevice.Ensure(host.size() * sizeof(unsigned char), "tokenmask dynamic class mask");
        CheckCuda(cudaMemcpy(mDynamicClassMaskDevice.As<unsigned char>(),
                             host.data(),
                             host.size() * sizeof(unsigned char),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy tokenmask dynamic class mask H2D");
        mDynamicClassMaskClasses = classCount;
    }

    cv::Mat DecodeDynamicMask(const OutputTensor &maskLogits,
                              const OutputTensor &classLogits) const
    {
        if(maskLogits.dims.size() < 4 || classLogits.dims.size() < 3)
            throw std::runtime_error("Unexpected TokenMask output rank");

        const int queryCount = maskLogits.dims[maskLogits.dims.size() - 3];
        const int maskHeight = maskLogits.dims[maskLogits.dims.size() - 2];
        const int maskWidth = maskLogits.dims[maskLogits.dims.size() - 1];
        const int classCount = classLogits.dims.back();

        if(queryCount <= 0 || maskHeight <= 0 || maskWidth <= 0 || classCount <= 1)
            throw std::runtime_error("Invalid TokenMask output shapes");
        if(maskLogits.Elements() < static_cast<int64_t>(queryCount) *
                                   static_cast<int64_t>(maskHeight) *
                                   static_cast<int64_t>(maskWidth) ||
           classLogits.Elements() < static_cast<int64_t>(queryCount) *
                                    static_cast<int64_t>(classCount))
            throw std::runtime_error("TokenMask output tensors are smaller than their decoded shapes");

        struct KeptQuery
        {
            int queryIndex;
            int classId;
            float score;
        };

        std::vector<KeptQuery> kept;
        kept.reserve(static_cast<size_t>(queryCount));
        const float *classPtr = classLogits.f32.data();
        const int noObjectClass = classCount - 1;

        for(int q = 0; q < queryCount; ++q)
        {
            const float *logits = classPtr + static_cast<size_t>(q) * static_cast<size_t>(classCount);
            float maxLogit = logits[0];
            int maxClass = 0;
            for(int c = 1; c < classCount; ++c)
            {
                if(logits[c] > maxLogit)
                {
                    maxLogit = logits[c];
                    maxClass = c;
                }
            }

            float expSum = 0.0f;
            for(int c = 0; c < classCount; ++c)
                expSum += std::exp(logits[c] - maxLogit);
            const float score = expSum > 0.0f ? 1.0f / expSum : 0.0f;

            if(maxClass != noObjectClass && score > mConfig.tokenmaskMaskThreshold)
                kept.push_back({q, maxClass, score});
        }

        cv::Mat dynamic(maskHeight, maskWidth, CV_8UC1, cv::Scalar(0));
        if(kept.empty())
            return dynamic;

        const size_t area = static_cast<size_t>(maskHeight) * static_cast<size_t>(maskWidth);
        const float *maskPtr = maskLogits.f32.data();
        std::vector<float> bestScores(area, -std::numeric_limits<float>::infinity());
        std::vector<int> bestQuery(area, -1);

        for(size_t k = 0; k < kept.size(); ++k)
        {
            const float *queryMask = maskPtr + static_cast<size_t>(kept[k].queryIndex) * area;
            for(size_t pixel = 0; pixel < area; ++pixel)
            {
                const float maskScore = kept[k].score * Sigmoid(queryMask[pixel]);
                if(maskScore > bestScores[pixel])
                {
                    bestScores[pixel] = maskScore;
                    bestQuery[pixel] = static_cast<int>(k);
                }
            }
        }

        for(size_t k = 0; k < kept.size(); ++k)
        {
            const float *queryMask = maskPtr + static_cast<size_t>(kept[k].queryIndex) * area;

            int originalArea = 0;
            int winnerArea = 0;
            int finalArea = 0;
            for(size_t pixel = 0; pixel < area; ++pixel)
            {
                const bool originalMask = Sigmoid(queryMask[pixel]) >= 0.5f;
                const bool winnerMask = bestQuery[pixel] == static_cast<int>(k);
                if(originalMask)
                    ++originalArea;
                if(winnerMask)
                    ++winnerArea;
                if(originalMask && winnerMask)
                    ++finalArea;
            }

            if(originalArea == 0 || winnerArea == 0 || finalArea == 0)
                continue;
            if(static_cast<float>(winnerArea) / static_cast<float>(originalArea) <
               mConfig.tokenmaskOverlapThreshold)
                continue;
            if(mDynamicLabels.find(kept[k].classId) == mDynamicLabels.end())
                continue;

            for(size_t pixel = 0; pixel < area; ++pixel)
            {
                const bool finalMask = Sigmoid(queryMask[pixel]) >= 0.5f &&
                                       bestQuery[pixel] == static_cast<int>(k);
                if(finalMask)
                    dynamic.ptr<unsigned char>(static_cast<int>(pixel / maskWidth))
                        [static_cast<int>(pixel % maskWidth)] = 255;
            }
        }

        return dynamic;
    }

    SemanticMaskerConfig mConfig;
    cv::Size mInputSize;
    std::string mDataset;
    TensorRTEngine mEngine;
    std::set<int> mDynamicLabels;
    bool mWarmingUp = false;
    int mDynamicClassMaskClasses = 0;
    CudaDeviceBuffer mInputImageDevice;
    CudaDeviceBuffer mDynamicClassMaskDevice;
    CudaDeviceBuffer mQueryClass;
    CudaDeviceBuffer mQueryScore;
    CudaDeviceBuffer mQueryKeep;
    CudaDeviceBuffer mBestQuery;
    CudaDeviceBuffer mOriginalArea;
    CudaDeviceBuffer mWinnerArea;
    CudaDeviceBuffer mFinalArea;
    CudaDeviceBuffer mQueryAccepted;
    CudaDeviceBuffer mDynamicSmall;
    CudaDeviceBuffer mFeatureMaskDevice;
};
#endif

std::unique_ptr<ISemanticBackend> CreateBackend(const SemanticMaskerConfig &config)
{
#ifdef LITESEMSLAM_USE_TENSORRT_SEMANTICS
    const std::string backend = ToLower(config.backend);
    if(backend == "segnet")
        return std::unique_ptr<ISemanticBackend>(new SegNetTensorRTBackend(config));
    if(backend == "maskrcnn" || backend == "mask_rcnn")
        return std::unique_ptr<ISemanticBackend>(new MaskRCNNTensorRTBackend(config));
    if(backend == "tokenmask" || backend == "token_mask")
        return std::unique_ptr<ISemanticBackend>(new TokenMaskTensorRTBackend(config));
    throw std::runtime_error("unsupported semantic backend: " + config.backend);
#else
    (void)config;
    throw std::runtime_error("LiteSemSLAM was built without TensorRT semantic support");
#endif
}
}

struct SemanticMasker::Impl
{
    explicit Impl(const SemanticMaskerConfig &cfg)
        : config(cfg)
    {
        const char *envDebugDir = std::getenv("LITESEMSLAM_SEMANTIC_DEBUG_DIR");
        if(envDebugDir && *envDebugDir)
            debugDir = envDebugDir;
    }

    ~Impl()
    {
        Stop();
    }

    bool Start()
    {
        if(!config.enabled)
            return false;

        {
            std::unique_lock<std::mutex> lock(mutex);
            if(running)
                return ready;
            stop = false;
            ready = false;
            failed = false;
        }

        worker = std::thread(&Impl::Run, this);

        std::unique_lock<std::mutex> lock(mutex);
        readyCv.wait(lock, [this](){ return ready || failed; });
        running = ready;
        return ready;
    }

    void Stop()
    {
        {
            std::unique_lock<std::mutex> lock(mutex);
            stop = true;
            running = false;
            requestCv.notify_all();
            resultCv.notify_all();
        }
        if(worker.joinable())
            worker.join();

        std::unique_lock<std::mutex> lock(mutex);
        running = false;
        ready = false;
    }

    bool IsEnabled() const
    {
        std::unique_lock<std::mutex> lock(mutex);
        return config.enabled && ready && running;
    }

    unsigned long long Submit(const cv::Mat &image, double timestamp)
    {
        unsigned long long frameIndex = 0;
        unsigned long long inferRequestId = 0;
        unsigned long long responseId = 0;
        bool reuseEnabled = false;
        bool shouldProcess = false;
        bool hasReusableMask = false;
        {
            std::unique_lock<std::mutex> lock(mutex);
            if(!ready || !running)
                return 0;

            frameIndex = ++submittedCount;
            reuseEnabled = config.frameStride > 1 || config.maxMaskAge > 0;
            shouldProcess = !reuseEnabled ||
                            latestMask.empty() ||
                            ((frameIndex - 1) % static_cast<unsigned long long>(config.frameStride) == 0);
            const int allowedAge = config.maxMaskAge;
            hasReusableMask = reuseEnabled &&
                              allowedAge > 0 &&
                              !latestMask.empty() &&
                              latestMaskFrameIndex > 0 &&
                              frameIndex >= latestMaskFrameIndex &&
                              static_cast<int>(frameIndex - latestMaskFrameIndex) <= allowedAge;

            if(shouldProcess)
                inferRequestId = ++nextId;

            if(hasReusableMask)
            {
                responseId = ++nextId;
                completedMasks[responseId] = latestMask;
            }
        }

        cv::Mat bgr;
        if(shouldProcess)
        {
            bgr = PrepareImageForBackend(image, config.inputIsRgb);
            if(bgr.empty())
                return responseId;
        }

        if(shouldProcess)
        {
            Request request;
            request.id = inferRequestId;
            request.frameIndex = frameIndex;
            request.timestamp = timestamp;
            request.image = bgr;
            request.storeExactResult = !reuseEnabled || !hasReusableMask;

            std::unique_lock<std::mutex> lock(mutex);
            if(!ready || !running)
                return responseId;
            if(reuseEnabled && hasReusableMask)
                requests.clear();
            requests.push_back(request);
            requestCv.notify_one();
        }

        if(hasReusableMask)
            return responseId;

        if(shouldProcess)
            return inferRequestId;

        return 0;
    }

    bool WaitForMask(unsigned long long requestId, cv::Mat &featureMask)
    {
        if(requestId == 0)
            return false;

        std::unique_lock<std::mutex> lock(mutex);
        const auto predicate = [this, requestId]() {
            return completedMasks.find(requestId) != completedMasks.end() ||
                   failedRequests.find(requestId) != failedRequests.end() ||
                   !running || failed;
        };

        resultCv.wait(lock, predicate);

        std::map<unsigned long long, cv::Mat>::iterator exact = completedMasks.find(requestId);
        if(exact != completedMasks.end())
        {
            featureMask = exact->second;
            completedMasks.erase(exact);
            return true;
        }

        failedRequests.erase(requestId);
        return false;
    }

    void Run()
    {
        std::unique_ptr<ISemanticBackend> localBackend;
        try
        {
            localBackend = CreateBackend(config);
        }
        catch(const std::exception &exc)
        {
            std::unique_lock<std::mutex> lock(mutex);
            failed = true;
            std::cerr << "SemanticMasker disabled: " << exc.what() << std::endl;
            readyCv.notify_all();
            resultCv.notify_all();
            return;
        }

        {
            std::unique_lock<std::mutex> lock(mutex);
            backend.swap(localBackend);
            ready = true;
            readyCv.notify_all();
            std::cout << "SemanticMasker TensorRT backend ready: " << config.backend << std::endl;
        }

        while(true)
        {
            Request request;
            {
                std::unique_lock<std::mutex> lock(mutex);
                requestCv.wait(lock, [this](){ return stop || !requests.empty(); });
                if(stop)
                    break;
                request = requests.front();
                requests.pop_front();
            }

            cv::Mat mask;
            try
            {
                backend->PredictFeatureMask(request.image, mask);
                if(!debugDir.empty())
                    DumpSemanticDebugFrame(debugDir, request.id, request.timestamp, request.image, mask);
            }
            catch(const std::exception &exc)
            {
                std::cerr << "SemanticMasker frame failed: " << exc.what() << std::endl;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    if(request.storeExactResult)
                        failedRequests.insert(request.id);
                    resultCv.notify_all();
                }
                continue;
            }

            {
                std::unique_lock<std::mutex> lock(mutex);
                latestMask = mask.clone();
                latestMaskFrameIndex = request.frameIndex;
                if(request.storeExactResult)
                    completedMasks[request.id] = mask;
                ++processedCount;
                resultCv.notify_all();
            }
        }
    }

    struct Request
    {
        unsigned long long id = 0;
        unsigned long long frameIndex = 0;
        double timestamp = 0.0;
        cv::Mat image;
        bool storeExactResult = true;
    };

    SemanticMaskerConfig config;
    std::string debugDir;
    mutable std::mutex mutex;
    std::condition_variable requestCv;
    std::condition_variable resultCv;
    std::condition_variable readyCv;
    std::thread worker;
    std::unique_ptr<ISemanticBackend> backend;
    std::deque<Request> requests;
    bool stop = false;
    bool running = false;
    bool ready = false;
    bool failed = false;
    unsigned long long nextId = 0;
    unsigned long long submittedCount = 0;
    unsigned long long processedCount = 0;
    unsigned long long latestMaskFrameIndex = 0;
    cv::Mat latestMask;
    std::set<unsigned long long> failedRequests;
    std::map<unsigned long long, cv::Mat> completedMasks;
};

SemanticMasker::SemanticMasker(const SemanticMaskerConfig &config)
    : mImpl(new Impl(config))
{
}

SemanticMasker::~SemanticMasker()
{
}

bool SemanticMasker::Start()
{
    return mImpl->Start();
}

void SemanticMasker::Stop()
{
    mImpl->Stop();
}

bool SemanticMasker::IsEnabled() const
{
    return mImpl->IsEnabled();
}

unsigned long long SemanticMasker::Submit(const cv::Mat &image, double timestamp)
{
    return mImpl->Submit(image, timestamp);
}

bool SemanticMasker::WaitForMask(unsigned long long requestId, cv::Mat &featureMask)
{
    return mImpl->WaitForMask(requestId, featureMask);
}

} // namespace ORB_SLAM3
