#include "cuda/synced_mem_holder.hpp"

#include<iostream>
#include<string.h>
#include<cstdlib>

namespace orb_cuda {

namespace {

bool g_use_cuda_host_memory = false;

template<typename Dtype>
void free_host_buffer(Dtype* data, bool use_cuda_host_memory)
{
    if(!data)
        return;

    if(use_cuda_host_memory)
        cudaFreeHost(data);
    else
        std::free(data);
}

template<typename Dtype>
Dtype* allocate_host_buffer(size_t count, bool use_cuda_host_memory)
{
    if(count == 0)
        return 0;

    Dtype* data = 0;
    if(use_cuda_host_memory)
        cudaHostAlloc(reinterpret_cast<void**>(&data), count * sizeof(Dtype), cudaHostAllocDefault);
    else
        data = static_cast<Dtype*>(std::malloc(count * sizeof(Dtype)));

    if(!data)
    {
        std::cerr << "host allocation failed for count " << count << std::endl;
        std::abort();
    }

    return data;
}

} // namespace

void ConfigureCudaHostMemory(bool enabled)
{
    g_use_cuda_host_memory = enabled;
}

bool CudaHostMemoryEnabled()
{
    return g_use_cuda_host_memory;
}

template<typename Dtype>
SyncedMem<Dtype>::SyncedMem(void)
{
    count_ = 0;
    capacity_ = 0;
    cpu_data_ = 0;
    gpu_data_ = 0;
    cu_stream_ = NULL;
    pitch_ = 0;
    use_cuda_host_memory_ = CudaHostMemoryEnabled();

    cudaStreamCreate(&cu_stream_);
}

template<typename Dtype>
SyncedMem<Dtype>::~SyncedMem() {

    //finish off anything going on in cu_stream_
    cudaStreamSynchronize(cu_stream_);
    cudaStreamDestroy(cu_stream_);

    cu_stream_ = NULL;
    free_host_buffer(cpu_data_, use_cuda_host_memory_);

    if(gpu_data_)
        cudaFree(gpu_data_);
}


template<typename Dtype>
void SyncedMem<Dtype>::resize_pitched(size_t width, size_t height)
{
    count_ = width * height;

    capacity_ = count_;

    free_host_buffer(cpu_data_, use_cuda_host_memory_);
    cpu_data_ = allocate_host_buffer<Dtype>(capacity_, use_cuda_host_memory_);

    if(gpu_data_)
        cudaFree(gpu_data_);
    cudaMallocPitch((void**)&gpu_data_, &pitch_, width * sizeof(Dtype), height);

}

template<typename Dtype>
void SyncedMem<Dtype>::resize(int count)
{
    count_ = count;

    if(capacity_ < count_)
    {
        capacity_ = count_;

        free_host_buffer(cpu_data_, use_cuda_host_memory_);
        cpu_data_ = allocate_host_buffer<Dtype>(capacity_, use_cuda_host_memory_);

        if(gpu_data_)
            cudaFree(gpu_data_);
        cudaMalloc((void**)&gpu_data_, capacity_ * sizeof(Dtype));

    }
}


template<typename Dtype>
Dtype* SyncedMem<Dtype>::cpu_data()
{
    return cpu_data_;
}

template<typename Dtype>
Dtype* SyncedMem<Dtype>::gpu_data()
{
    return gpu_data_;
}


template<typename Dtype>
void SyncedMem<Dtype>::to_cpu(void)
{
    cudaMemcpy(cpu_data_, gpu_data_, count_ * sizeof(Dtype), cudaMemcpyDeviceToHost);
}

template<typename Dtype>
void SyncedMem<Dtype>::to_gpu(void)
{
    cudaMemcpy(gpu_data_, cpu_data_, count_ * sizeof(Dtype), cudaMemcpyHostToDevice);
}


template<typename Dtype>
void SyncedMem<Dtype>::set_zero_cpu(void)
{
    memset(cpu_data_, 0, count_ * sizeof(Dtype));
}

template<typename Dtype>
void SyncedMem<Dtype>::set_zero_gpu(void)
{
    cudaMemset(gpu_data_, 0, count_ * sizeof(Dtype));
}

template<typename Dtype>
void SyncedMem<Dtype>::set_zero_gpu_async(void)
{
    cudaMemsetAsync(gpu_data_, 0, count_ * sizeof(Dtype), cu_stream_);
}


template<typename Dtype>
void SyncedMem<Dtype>::to_cpu_async(void)
{
    cudaMemcpyAsync(cpu_data_, gpu_data_, count_ * sizeof(Dtype), cudaMemcpyDeviceToHost, cu_stream_);
}

template<typename Dtype>
void SyncedMem<Dtype>::to_gpu_async(void)
{
    cudaMemcpyAsync(gpu_data_, cpu_data_, count_ * sizeof(Dtype), cudaMemcpyHostToDevice, cu_stream_);
}

template<typename Dtype>
void SyncedMem<Dtype>::to_cpu_async(cudaStream_t& cu_stream)
{
    cudaMemcpyAsync(cpu_data_, gpu_data_, count_ * sizeof(Dtype), cudaMemcpyDeviceToHost, cu_stream);
}

template<typename Dtype>
void SyncedMem<Dtype>::to_gpu_async(cudaStream_t& cu_stream)
{
    cudaMemcpyAsync(gpu_data_, cpu_data_, count_ * sizeof(Dtype), cudaMemcpyHostToDevice, cu_stream);
}


// desired count transfer CPU-GPU
template<typename Dtype>
void SyncedMem<Dtype>::to_cpu(int count)
{
    cudaMemcpy(cpu_data_, gpu_data_, count * sizeof(Dtype), cudaMemcpyDeviceToHost);
}

template<typename Dtype>
void SyncedMem<Dtype>::to_gpu(int count)
{
    cudaMemcpy(gpu_data_, cpu_data_, count * sizeof(Dtype), cudaMemcpyHostToDevice);
}



template<typename Dtype>
void SyncedMem<Dtype>::to_cpu_async(int count)
{
    cudaMemcpyAsync(cpu_data_, gpu_data_, count * sizeof(Dtype), cudaMemcpyDeviceToHost, cu_stream_);
}

template<typename Dtype>
void SyncedMem<Dtype>::to_gpu_async(int count)
{
    cudaMemcpyAsync(gpu_data_, cpu_data_, count * sizeof(Dtype), cudaMemcpyHostToDevice, cu_stream_);
}

template<typename Dtype>
void SyncedMem<Dtype>::to_cpu_async(cudaStream_t& cu_stream,int count)
{
    cudaMemcpyAsync(cpu_data_, gpu_data_, count * sizeof(Dtype), cudaMemcpyDeviceToHost, cu_stream);
}

template<typename Dtype>
void SyncedMem<Dtype>::to_gpu_async(cudaStream_t& cu_stream,int count)
{
    cudaMemcpyAsync(gpu_data_, cpu_data_, count * sizeof(Dtype), cudaMemcpyHostToDevice, cu_stream);
}



template<typename Dtype>
void SyncedMem<Dtype>::sync_stream(void)
{
    cudaStreamSynchronize(cu_stream_);
}

template class SyncedMem<unsigned char>;
template class SyncedMem<signed char>;
template class SyncedMem<unsigned short>;
template class SyncedMem<short>;
template class SyncedMem<unsigned int>;
template class SyncedMem<int>;
template class SyncedMem<float>;


}



