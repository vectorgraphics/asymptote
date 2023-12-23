#pragma once

#ifndef __INTELLISENSE__
#ifndef KERNEL_ARGS
#define KERNEL_ARGS(blk,thrdsz) <<<blk,thrdsz>>>
#endif
#else
#define KERNEL_ARGS(blk,thrdsz)
#define __CUDACC__
#include <device_functions.h>
#endif

#include <cuda_runtime.h>

void map_reflectance_ker(float4* in, float3* out, size_t width, size_t height, float roughness, size_t outWidth, size_t outHeight);
void generate_brdf_integrate_lut_ker(int width, int height, float2* out);
