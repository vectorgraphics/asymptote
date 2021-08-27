/**
* @file kernel.cu
* @author Supakorn "Jamie" Rassameemasmuang <jamievlin@outlook.com>
* CUDA Kernel for computing irradiance by solid angle integration
*/

#include "kernel.h"
#include "helper.cuh"
#include "linalg.cuh"

#include <cuda.h>
#include <texture_indirect_functions.h>
#include <device_launch_parameters.h>

// Can we encode this somewhere else?
__device__ constexpr int PHI_SAMPLES = 300;
//__device__ constexpr int n = PHI_SAMPLES/2;
__device__ constexpr int THETA_SAMPLES = 400;

__device__ constexpr float THETA_INTEGRATION_REGION = HALFPI;
__device__ constexpr float PHI_INTEGRATION_REGION = TAU;
__device__ constexpr float dx_int_scale =
    (0.5 * THETA_INTEGRATION_REGION * PHI_INTEGRATION_REGION) / (PI * PHI_SAMPLES * THETA_SAMPLES);

// #define TEST_NO_INTEGRAL



__device__
float3 inner(float sampled_phi, const float3 N, const float3 N1, const float3 N2,
           cudaTextureObject_t tObjin, size_t width, size_t height)
{
  float3 sum3=make_float3(0,0,0);
  for (int j = 0; j < THETA_SAMPLES; ++j)
    {
      float sampled_theta = j * THETA_INTEGRATION_REGION / THETA_SAMPLES;

// vec3 is the world space coordinate
      float2 sphcoord = to_sphcoord(angleToBasis(N, N1, N2, sampled_phi, sampled_theta));

      float4 frag = tex2D<float4>(tObjin,
                                  sphcoord.x * PI_RECR * width / 2,
                                  sphcoord.y * PI_RECR * height);

      float scale = __sinf(2*sampled_theta);
      float3 frag3=make_float3(frag.x,frag.y,frag.z);
      float3_addinplace(sum3,frag3,scale);
    }
  return sum3;
}

__global__
void irradiate(cudaTextureObject_t tObjin, float3* out, size_t width, size_t height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;


    if (idx < width && idx_y < height)
    {
        int access_idx = to_idx(width, idx, idx_y);

        out[access_idx] = make_float3(0, 0, 0);

        float target_phi = TAU * ((idx + 0.5f) / width);
        float target_theta = PI * ((idx_y + 0.5f) / height);

        const float3 N = from_sphcoord(target_phi, target_theta);
        const float3 N1 = make_float3(
            __cosf(target_theta) * __cosf(target_phi),
            __cosf(target_theta) * __sinf(target_phi),
            -1*__sinf(target_theta));
        const float3 N2 = make_float3(-1 * __sinf(target_phi), __cosf(target_phi), 0);

#ifndef TEST_NO_INTEGRAL
        float3 sum3=make_float3(0,0,0);
        for (int i = 0; i < PHI_SAMPLES; ++i)
        {
            float sampled_phi = i * PHI_INTEGRATION_REGION / PHI_SAMPLES;

            float3 inside=inner(sampled_phi,N,N1,N2,tObjin,width,height);
            float3_addinplace(sum3,inside);
        }

        float3_addinplace(out[access_idx], sum3, dx_int_scale);

#else
        float2 sphcoord = to_sphcoord(angleToBasis(N, N1, N2, 0, 0));

        float4 frag = tex2D<float4>(tObjin,
            sphcoord.x * PI_RECR * width / 2,
            sphcoord.y * PI_RECR * height);

        float3 frag3 = make_float3(frag.x, frag.y, frag.z);
        out[access_idx] = frag3;
#endif
    }
}

const size_t blkSz = 15;
void irradiate_ker(float4* in, float3* out, size_t width, size_t height)
{
    float4* d_ptr;
    size_t pitch;
    cudaErrorCheck(cudaMallocPitch(
        &d_ptr, &pitch, width * sizeof(float4), height));
    cudaErrorCheck(cudaMemcpy2D(d_ptr, pitch, in,
        width * sizeof(float4), width*sizeof(float4),
        height, cudaMemcpyHostToDevice));

    cudaResourceDesc cRD;
    memset(&cRD, 0, sizeof(cudaResourceDesc));
    cRD.resType = cudaResourceTypePitch2D;
    cRD.res.pitch2D.devPtr = d_ptr;
    cRD.res.pitch2D.width = width;
    cRD.res.pitch2D.height = height;
    cRD.res.pitch2D.desc = cudaCreateChannelDesc<float4>();
    cRD.res.pitch2D.pitchInBytes = pitch;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.sRGB = 0;
    texDesc.readMode = cudaReadModeElementType;

    cudaTextureObject_t t_obj;
    cudaErrorCheck(cudaCreateTextureObject(
        &t_obj, &cRD, &texDesc, nullptr));

    // out source
    float3* d_out;
    cudaErrorCheck(cudaMalloc(
        (void**)&d_out, static_cast<size_t>(sizeof(float3) * width * height)));
    dim3 blockSz((width / blkSz) + 1, (height / blkSz) + 1);
    dim3 kerSz(blkSz, blkSz);
    irradiate KERNEL_ARGS(blockSz, kerSz) (t_obj, d_out, width, height);

    cudaErrorCheck(cudaMemcpy(
        out, d_out, sizeof(float3) * width * height, cudaMemcpyDeviceToHost));
    cudaErrorCheck(cudaDestroyTextureObject(t_obj));
    cudaFree(d_ptr);
}
