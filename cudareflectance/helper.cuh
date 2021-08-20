#pragma once

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

inline void check_error()
{
#ifdef DEBUG
    cudaError_t __err = cudaGetLastError();
    if (__err != cudaSuccess)
    {
        std::cerr << "Fata CUDA Error. Msg: " << cudaGetErrorString(__err) << std::endl;
    }
    else
    {
        std::cerr << "CUDA Operation success" << std::endl;
    }
#endif
}

inline void cudaErrorCheck(cudaError err)
{
    if (err != cudaSuccess)
    {
        std::cerr << "ERROR: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

template<typename T>
T* cudaDupe(T* in, size_t sz)
{
    T* out = nullptr;
    cudaErrorCheck(cudaMalloc((void**)&out, sizeof(T) * sz));
    cudaErrorCheck(cudaMemcpy(out, in, sizeof(T) * sz, cudaMemcpyHostToDevice));
    return out;
}

__device__
inline int to_idx(int width, int x, int y)
{
    return y * width + x;
}
