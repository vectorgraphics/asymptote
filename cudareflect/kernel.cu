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
__device__ constexpr int HALF_PHI_SAMPLES = 150;
__device__ constexpr int THETA_SAMPLES = 400;
__device__ constexpr float hPHI=PI/HALF_PHI_SAMPLES;

__device__ constexpr float THETA_INTEGRATION_REGION = HALFPI;
__device__ constexpr float dx_int_scale =
    (0.5 * THETA_INTEGRATION_REGION * hPHI) / (3.0*PI * THETA_SAMPLES);

class IntegrateSampler
{
public:
    __device__
    IntegrateSampler(
        cudaTextureObject_t tObjin,
        float3 const& n, float3 const& n1, float3 const& n2,
        size_t const& inWidth, size_t const& inHeight) :
        N(n), N1(n1), N2(n2), width(inWidth), height(inHeight),
        tObj(tObjin)

    {
    }

    __device__ ~IntegrateSampler() {}

    __device__
    float3 integrand(float const& sampled_phi, float const& sampled_theta)
    {
        // vec3 is the world space coordinate
        float2 sphcoord = to_sphcoord(angleToBasis(N, N1, N2, sampled_phi, sampled_theta));

        float4 frag = tex2D<float4>(tObj,
            sphcoord.x * PI_RECR * width / 2,
            sphcoord.y * PI_RECR * height);
        return make_float3(frag.x, frag.y, frag.z);
    }

    __device__
    float3 inner(float sampled_phi)
    {
        float3 sum3 = make_float3(0, 0, 0);
        for (int j = 0; j < THETA_SAMPLES; ++j)
        {
            float sampled_theta = j * THETA_INTEGRATION_REGION / THETA_SAMPLES;

            float3 result = integrand(sampled_phi, sampled_theta);

            // 2*sin(sampled_theta)*cos(sampled_theta). The "times 2" part is cancelled
            // out in dx_int_scale.
            float scale = __sinf(2 * sampled_theta);
            float3_addinplace(sum3, result, scale);
        }
        return sum3;
    }

    __device__
    float3 integrate()
    {
        float3 out=inner(0.0);
        float3 sumeven=make_float3(0,0,0);
        unsigned int n=2*HALF_PHI_SAMPLES;
        for (int i = 2; i < n; i += 2)
          float3_addinplace(sumeven,inner(i*hPHI));
        float3_addinplace(out,sumeven,2.0);
        float3 sumodd=make_float3(0,0,0);
        for (int i = 2; i < n; i += 2)
          float3_addinplace(sumodd,inner((i-1)*hPHI));
        float3_addinplace(out,sumodd,4.0);
        float3_addinplace(out,inner(TAU));
        float3_scaleinplace(out,dx_int_scale);
        return out;
    }

private:
    float3 N, N1, N2;
    size_t width, height;
    cudaTextureObject_t tObj;
};



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


        IntegrateSampler integrator(tObjin, N, N1, N2, width, height);
        out[access_idx]=integrator.integrate();
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
