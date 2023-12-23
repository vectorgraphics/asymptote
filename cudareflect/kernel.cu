/**
* @file kernel.cu
* @author Supakorn "Jamie" Rassameemasmuang <jamievlin@outlook.com>
* CUDA Kernel for computing irradiance by solid angle integration

* Partially based on:
* https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf
*/

#include "kernel.h"
#include "helper.cuh"
#include "utils.cuh"

#include "simpson.cuh"

#include <cuda.h>
#include <texture_indirect_functions.h>
#include <device_launch_parameters.h>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <functional>

class IntegrateSampler
{
public:
    __device__
    IntegrateSampler(
        cudaTextureObject_t tObjin,
        glm::mat3 normalOrthBasis,
        size_t const& inWidth, size_t const& inHeight) :
        normalOrthBasis(normalOrthBasis), width(inWidth), height(inHeight),
        tObj(tObjin)

    {
    }

    __device__ ~IntegrateSampler() {}

    __device__
    glm::vec3 integrand(float const& sampled_phi, float const& sampled_theta)
    {
        // vec3 is the world space coordinate
        glm::vec2 sphcoord = to_sphcoord(angleToBasis(normalOrthBasis, sampled_phi, sampled_theta));
        float4 frag = tex2D<float4>(tObj,
            sphcoord.x * PI_RECR * 0.5*width,
            sphcoord.y * PI_RECR * height);

        return glm::vec3(frag.x, frag.y, frag.z);
    }

    __device__
    glm::vec3 inner(float const& sampled_theta)
    {
        return simpson(
          [this, &sampled_theta](float const& phi) {return this->integrand(phi,sampled_theta);  },
          0, TAU, acc)*0.5f*__sinf(2 * sampled_theta);
    }

    __device__
    glm::vec3 integrate()
    {
        return PI_RECR * simpson(
            [this](float const& theta) {return this->inner(theta); },
            0, HALFPI, acc);
    }

private:
    glm::mat3 normalOrthBasis;
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

        float target_phi = TAU * ((idx + 0.5f) / width);
        float target_theta = PI * ((idx_y + 0.5f) / height);

        const glm::vec3 N = from_sphcoord(target_phi, target_theta);
        const glm::vec3 N1(
            __cosf(target_theta) * __cosf(target_phi),
            __cosf(target_theta) * __sinf(target_phi),
            -1*__sinf(target_theta));
        const glm::vec3 N2(-1 * __sinf(target_phi), __cosf(target_phi), 0);

        glm::mat3 normalBasisMat(N1,N2,N);

        IntegrateSampler integrator(tObjin, normalBasisMat, width, height);
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
