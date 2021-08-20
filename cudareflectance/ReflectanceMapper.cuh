#pragma once

#ifndef __INTELLISENSE__
#ifndef KERNEL_ARGS
#define KERNEL_ARGS(blk,thrdsz) <<<blk,thrdsz>>>
#endif
#else
#define KERNEL_ARGS(blk,thrdsz)
#define __CUDACC__
#endif

#include <cuda_runtime.h>

void map_reflectance_ker(float4* in, float3* out, size_t width, size_t height, float roughness);