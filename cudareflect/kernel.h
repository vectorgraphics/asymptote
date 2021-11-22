/**
* @file kernel.h
* @author Supakorn "Jamie" Rassameemasmuang <jamievlin@outlook.com>
* CUDA Kernel Header for computing irradiance by solid angle integration
*/
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
void irradiate_ker(float4* in, float3* out, size_t width, size_t height);
