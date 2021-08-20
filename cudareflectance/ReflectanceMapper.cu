#include "ReflectanceMapper.cuh"

#include "helper.cuh"
#include "linalg.cuh"

#include <cuda.h>
#include <device_launch_parameters.h>

__device__
inline float swap_bits(uint32_t const& x, uint32_t const& mask_1, unsigned int const& shft)
{
    return ((x & mask_1) << shft) | ((x & (~mask_1)) >> shft);
}

__device__ constexpr float recvbit = 2.32830643654e-10; // 1/2^32.
__device__ constexpr int REFL_NUM_SAMPLES = 1 << 15;
__device__ constexpr float INV_REFL_NUM_SAMPLES = 1.f / REFL_NUM_SAMPLES;

__device__
float van_der_corput_bitshift(uint32_t bits)
{
    bits = swap_bits(bits, 0x55555555, 1);
    bits = swap_bits(bits, 0x33333333, 2);
    bits = swap_bits(bits, 0x0F0F0F0F, 4);
    bits = swap_bits(bits, 0x00FF00FF, 8);
    bits = swap_bits(bits, 0x0000FFFF, 16);

    return static_cast<float>(bits) * recvbit;
}

__device__
float2 hamersely(uint32_t i, uint32_t N)
{
    return make_float2(static_cast<float>(i) / N, van_der_corput_bitshift(i));
}

__device__
float3 importance_sampl_GGX(float2 sample, float3 normal, float roughness)
{
    float a = roughness * roughness;

    float phi = TAU * sample.x;
    float cosTheta = sqrtf((1.0f - sample.y) / (1.f + (a * a - 1.f) * sample.y)); // GGX Sample, inverse sampling?
    // TODO: Understand the derivation behind this cosTheta. It has something to do with GGX distribution, but how?
    float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);

    float3 vec = from_sphcoord(phi, cosTheta, sinTheta);
    float3 N1 = make_float3(
        cosTheta * __cosf(phi),
        cosTheta * __sinf(phi),
        -1 * sinTheta);
    float3 N2 = make_float3(-1 * __sinf(phi), __cosf(phi), 0);

    return matrix3_multiply(N1, N2, normal, vec);
}

__global__
void map_reflectance(cudaTextureObject_t tObj, int width, int height, float roughness, float3* out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < width && idx_y < height)
    {
        int access_idx = to_idx(width, idx, idx_y);
        out[access_idx] = make_float3(0, 0, 0);

        float target_phi = TAU * ((idx + 0.5f) / width);
        float target_theta = PI * ((idx_y + 0.5f) / height);
        float3 N = from_sphcoord(target_phi, target_theta);

        float total_weight = 0.0f;
        for (int i = 0; i < REFL_NUM_SAMPLES; ++i)
        {
            float2 sample = hamersely(i, REFL_NUM_SAMPLES);
            float3 half_vec = importance_sampl_GGX(sample, N, roughness);


            // use the structure of parallelograms to calculate lightvec
            float3 scaled_half = float3_scale(half_vec, 2 * float3_dot(half_vec, N));
            float3 lightvec_raw = float3_subtract(scaled_half, N);
            float3 lightvec = float3_scale(lightvec_raw, rnorm3df(lightvec_raw.x, lightvec_raw.y, lightvec_raw.z));

            float ndotl = float3_dot(lightvec, N);
            float2 sphcoord = to_sphcoord(lightvec);
            float4 frag = tex2D<float4>(tObj,
                sphcoord.x * PI_RECR * width / 2,
                sphcoord.y * PI_RECR * height);

            float3 frag3 = make_float3(frag.x, frag.y, frag.z);

            if (ndotl > 0.0)
            {
                // epic games said it gives better results, otherwise weight can be set to 1.
#ifndef SET_WEIGHT_ONE
                float weight = ndotl;
#else
                float weight = 1.0f;
#endif
                float3_addinplace(out[access_idx], frag3, weight); // weighting by n@l, technically not required,

                total_weight += weight;
            }

        }

        if (total_weight > 0.0f)
            out[access_idx] = float3_scale(out[access_idx], 1 / total_weight);
        else
            out[access_idx] = make_float3(0, 0, 0);

    }
}

const size_t blkSz = 15;
void map_reflectance_ker(float4* in, float3* out, size_t width, size_t height, float roughness)
{
    float4* d_ptr;
    size_t pitch;
    cudaErrorCheck(cudaMallocPitch(
        &d_ptr, &pitch, width * sizeof(float4), height));
    cudaErrorCheck(cudaMemcpy2D(d_ptr, pitch, in,
        width * sizeof(float4), width * sizeof(float4),
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
    float3* d_out = nullptr;
    cudaErrorCheck(cudaMalloc(
        (void**)&d_out, static_cast<size_t>(sizeof(float3) * width * height)));
    dim3 blockSz((width / blkSz) + 1, (height / blkSz) + 1);
    dim3 kerSz(blkSz, blkSz);
    map_reflectance KERNEL_ARGS(blockSz, kerSz) (t_obj, width, height, roughness, d_out);

    cudaErrorCheck(cudaMemcpy(
        out, d_out, sizeof(float3) * width * height, cudaMemcpyDeviceToHost));

    cudaErrorCheck(cudaDestroyTextureObject(t_obj));
    cudaFree(d_ptr);
}