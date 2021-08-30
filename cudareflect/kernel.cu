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

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <functional>

// Can we encode this somewhere else?
__device__ constexpr int HALF_PHI_SAMPLES = 300;
__device__ constexpr int HALF_THETA_SAMPLES = 400;

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
    glm::vec3 integrand(float const& sampled_phi, float const& sampled_theta)
    {
        // vec3 is the world space coordinate
        float2 sphcoord = to_sphcoord(angleToBasis(N, N1, N2, sampled_phi, sampled_theta));
        float4 frag = tex2D<float4>(tObj,
            sphcoord.x * PI_RECR * width / 2,
            sphcoord.y * PI_RECR * height);

        float scale=__sinf(2 * sampled_theta);
        return glm::vec3(frag.x, frag.y, frag.z) * 0.5f * scale;
    }

    __device__
    glm::vec3 inner(float const& sampled_phi)
    {
        return simpsonThird(
            [this, &sampled_phi](float const& theta) {return this->integrand(sampled_phi, theta);  },
            0, HALFPI, HALF_THETA_SAMPLES);
    }

    __device__
    glm::vec3 integrate()
    {
        return PI_RECR * simpsonThird(
            [this](float const& ft) {return this->inner(ft); },
            0, TAU, HALF_PHI_SAMPLES);
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
        glm::vec3 out_val = integrator.integrate();
        out[access_idx] = make_float3(out_val.x, out_val.y, out_val.z);
    }
}

const size_t blkSz = 8;
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
