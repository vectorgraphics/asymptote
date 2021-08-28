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


#pragma region mapReflectance
__global__
void map_reflectance(cudaTextureObject_t tObj,
    int width, int height, float roughness,
    float3* out, int outWidth, int outHeight)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < outWidth && idx_y < outHeight)
    {
        int access_idx = to_idx(outWidth, idx, idx_y);
        out[access_idx] = make_float3(0, 0, 0);

        float target_phi = TAU * ((idx + 0.5f) / outWidth);
        float target_theta = PI * ((idx_y + 0.5f) / outHeight);
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
void map_reflectance_ker(
    float4* in, float3* out, size_t width, size_t height, float roughness,
    size_t outWidth, size_t outHeight)
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
        (void**)&d_out, static_cast<size_t>(sizeof(float3) * outWidth * outHeight)));
    dim3 blockSz((outWidth / blkSz) + 1, (outHeight / blkSz) + 1);
    dim3 kerSz(blkSz, blkSz);
    map_reflectance KERNEL_ARGS(blockSz, kerSz) (t_obj, width, height, roughness, d_out, outWidth, outHeight);

    cudaErrorCheck(cudaMemcpy(
        out, d_out, sizeof(float3) * outWidth * outHeight, cudaMemcpyDeviceToHost));

    cudaErrorCheck(cudaDestroyTextureObject(t_obj));
    cudaErrorCheck(cudaFree(d_ptr));
}


#pragma endregion

__device__
float norm_dist(float const& roughness, float3 const& half_vec)
{
    float alpha = roughness * roughness;
    float ndoth = half_vec.z; // assume N=(0,0,1)
    float denom_base = (ndoth * ndoth) * (alpha * alpha - 1) + 1;
    return (alpha * alpha) / (denom_base * denom_base * PI);
}

__device__
float G_component(float const& k, float const& ndotv)
{
    float denom = (ndotv * (1 - k)) + k;
    return 1 / denom;
}

__device__
float GFn(float const& roughness, float const& ndotl, float const& ndotv)
{
    float a = roughness * roughness;
    float k = a * a * 0.5;
    return G_component(k, ndotl) * G_component(k, ndotv);
}

__device__
float clamp(float const& x)
{
    return fminf(fmaxf(x, 0.0f), 1.0f);
}

// by symmetry, assume phi_v=0.
// porting of python code to CUDA

__device__ constexpr int LUT_INTEGRATE_SAMPLES = 8192;
__device__ constexpr float INTEGRATE_LUT_SCALE = 1.0f / LUT_INTEGRATE_SAMPLES;

__device__
float2 get_integrate_value(float const& roughness, float const& cos_theta)
{
    float2 value = make_float2(0, 0);
    float3 upZ = make_float3(0, 0, 1.0f);
    float num_samples = 0.0f;

    float cos_theta_v = clamp(cos_theta);
    float sin_theta_v = sqrtf(1 - cos_theta_v * cos_theta_v);
    float3 view_vec = make_float3(sin_theta_v, 0, cos_theta_v);

    for (int i=0; i< LUT_INTEGRATE_SAMPLES; ++i)
    {
        float2 sample_coord = hamersely(i, LUT_INTEGRATE_SAMPLES);
        float3 half_vec = importance_sampl_GGX(sample_coord, upZ, roughness);

        float3 scaled_half = float3_scale(half_vec, 2 * float3_dot(view_vec, half_vec));
        float3 lightvec_raw = float3_subtract(scaled_half, view_vec);
        float3 lightvec = float3_scale(lightvec_raw, rnorm3df(lightvec_raw.x, lightvec_raw.y, lightvec_raw.z));

        float ldotn = clamp(lightvec.z);
        float vdoth = clamp(float3_dot(half_vec, view_vec));
        float ndoth = clamp(half_vec.z);

        if (ldotn > 0.0f)
        {
            float base_val = (GFn(roughness, ldotn, cos_theta_v) * cos_theta_v * ldotn);
            float base_f = powf(1.0f - vdoth, 5.0f);
            
            value.x += base_val * (1 - base_f);
            value.y += base_val * base_f;
            num_samples += 1.0f;
        }
    }
    value.x = value.x * INTEGRATE_LUT_SCALE;
    value.y = value.y * INTEGRATE_LUT_SCALE;

    return value;
}

__global__
void generate_brdf_integrate(int width, int height, float2* out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < width && idx_y < height)
    {
        int access_idx = to_idx(width, idx, idx_y);
        float cosv = (idx + 1.0f) / width;
        float roughness = (idx_y + 1.0f) / height;

        out[access_idx] = get_integrate_value(roughness, cosv);
    }
}

void generate_brdf_integrate_lut_ker(int width, int height, float2* out)
{
    float2* d_out;
    cudaErrorCheck(cudaMalloc(
        (void**)&d_out, static_cast<size_t>(width * height * sizeof(float2))));


    dim3 blockSz((width / blkSz) + 1, (height / blkSz) + 1);
    dim3 kerSz(blkSz, blkSz);
    generate_brdf_integrate KERNEL_ARGS(blockSz, kerSz) (width, height, d_out);

    cudaErrorCheck(cudaMemcpy(
        out, d_out, width * height * sizeof(float2), cudaMemcpyDeviceToHost));
    cudaErrorCheck(cudaFree(d_out));

}