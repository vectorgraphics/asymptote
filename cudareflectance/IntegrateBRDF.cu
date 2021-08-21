#include "IntegrateBRDF.cuh"
#include "helper.cuh"
#include "linalg.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

constexpr int INTEGRATE_SAMPLE_PHI = 256;
constexpr int INTEGRATE_SAMPLE_THETA = 2048;

constexpr float INV_INTEGRATE_SAMPLE_PHI = 1.0f/INTEGRATE_SAMPLE_PHI;
constexpr float INV_INTEGRATE_SAMPLE_THETA = 1.0f/INTEGRATE_SAMPLE_THETA;
constexpr float SCALE = INV_INTEGRATE_SAMPLE_PHI * INV_INTEGRATE_SAMPLE_THETA;

__device__
float norm_dist(float const& roughness, float3 const& half_vec)
{
    float alpha = roughness * roughness;
    float ndoth = half_vec.z; // assume N=(0,0,1)
    float denom_base = (ndoth * ndoth) * (alpha * alpha - 1) + 1;
    return (alpha * alpha) / (denom_base * denom_base * PI);
}

__device__
float G_component(float k, float3 view)
{
    float ndotv = view.z;
    float denom = ndotv * (1 - k) + k;
    return ndotv / denom;
}

__device__
float GFn(float roughness, float3 lightvec, float3 viewvec)
{
    float r2 = roughness * roughness;
    float k = ((r2 + 1) * (r2 + 1)) / 8;
    return G_component(k, lightvec) * G_component(k, viewvec);
}

// by symmetry, assume phi_v=0.
// porting of python code to CUDA
__device__
float2 reflectance_integrand(float const& roughness, float const& cos_theta_v, float const& phi_l, float const& theta_l)
{
    float sin_theta_v = sqrtf(1 - cos_theta_v * cos_theta_v);
    float3 view_vec = make_float3(0, sin_theta_v, cos_theta_v);
    float3 light_vec = from_sphcoord(phi_l, theta_l);

    float3 half_vec_raw = float3_add(light_vec, view_vec);
    float3 half_vec = float3_scale(half_vec_raw, rnorm3df(half_vec_raw.x, half_vec_raw.y, half_vec_raw.z));

    float base_val = (
        norm_dist(roughness, half_vec) * GFn(roughness, light_vec, view_vec) * __cosf(theta_l))
        / (4 * half_vec.z * cos_theta_v);

    float base_f = powf(1.0f - float3_dot(view_vec, half_vec),5);
    return make_float2(
        base_val * (1 - base_f),
        base_val * base_f);
}

// FIXME: Can use Hammersly+GGX importance sampling instead of smapling uniformly.
// can improve the convergence of integral.
__device__
float2 get_integrate_value(float const& roughness, float const& cos_theta)
{
    float2 value = make_float2(0, 0);
    for (int i = 0; i < INTEGRATE_SAMPLE_PHI; ++i)
    {
        for (int j = 0; j < INTEGRATE_SAMPLE_THETA; ++j)
        {
            float target_phi = TAU * (i + 0.5f) * INV_INTEGRATE_SAMPLE_PHI;
            float target_theta = HALFPI * (j + 0.5f) * INV_INTEGRATE_SAMPLE_THETA;
            
            float2 intvalue = reflectance_integrand(roughness, cos_theta, target_phi, target_theta);
            float scale = SCALE *__sinf(target_theta);
            value.x += (intvalue.x * scale);
            value.y += (intvalue.y * scale);
        }
    }

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
        float roughness = (idx+0.5f) / width;
        float cosv = (idx_y+0.5f) / height;

        out[access_idx] = get_integrate_value(roughness, cosv);
    }
}

static constexpr int blkSz = 15;

void generate_brdf_integrate_ker(int width, int height, float2* out)
{
    float2* d_out;
    cudaErrorCheck(cudaMalloc(
        (void**)&d_out, static_cast<size_t>(width * height * sizeof(float2))));


    dim3 blockSz((width / blkSz) + 1, (height / blkSz) + 1);
    dim3 kerSz(blkSz, blkSz);
    generate_brdf_integrate KERNEL_ARGS(blockSz,kerSz) (width, height, d_out);

    cudaErrorCheck(cudaMemcpy(
        out, d_out, width * height * sizeof(float2), cudaMemcpyDeviceToHost));
    cudaErrorCheck(cudaFree(d_out));

}