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

void generate_brdf_integrate_ker(int width, int height, float2* out);