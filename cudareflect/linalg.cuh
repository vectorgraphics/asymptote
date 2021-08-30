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

#include <cuda.h>
#include <cuda_runtime.h>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

__device__ constexpr float PI = 3.141592654;
__device__ constexpr float HALFPI = 0.5*PI;
__device__ constexpr float TAU = 2.0*PI;
__device__ constexpr float PI_RECR = 1.0/PI;

__device__
inline float3 float3_add(float3 const& f1, float3 const& f2)
{
    return make_float3(f1.x + f2.x, f1.y + f2.y, f1.z + f2.z);
}

__device__
inline float float3_dot(float3 const& f1, float3 const& f2)
{
    return (f1.x * f2.x) + (f1.y * f2.y) + (f1.z * f2.z);
}

__device__
inline float3 float3_scale(float3 const& f1, float const& s1)
{
    return make_float3(f1.x * s1, f1.y * s1, f1.z * s1);
}

__device__
inline float3 float3_subtract(float3 const& f1, float3 const& f2)
{
    return make_float3(f1.x - f2.x, f1.y - f2.y, f1.z - f2.z);
}

/**
* @returns the result M*vec, where M is the matrix with
* $$
* M = (V_1 V_2 V_3)
* $$
* where V_1, V_2, V_3 are the three vectors in column form.
*/
__device__
inline float3 matrix3_multiply(float3 const& V1, float3 const& V2, float3 const& V3, float3 const& vec)
{
    return make_float3(
        float3_dot(make_float3(V1.x, V2.x, V3.x), vec),
        float3_dot(make_float3(V1.y, V2.y, V3.y), vec),
        float3_dot(make_float3(V1.z, V2.z, V3.z), vec)
    );
}

__device__
inline float2 to_sphcoord(float3 const& vec)
{
    return make_float2(
        atan2f(-vec.y, -vec.x) + PI,
        acosf(vec.z)
    );
}

__device__
inline float3 from_sphcoord(float const& phi, float const& theta)
{
    return make_float3(
        __sinf(theta) * __cosf(phi),
        __sinf(theta) * __sinf(phi),
        __cosf(theta));
}

__device__
inline float3 from_sphcoord(float const& phi, float const& cosTheta, float const& sinTheta)
{
    return make_float3(
        sinTheta * __cosf(phi),
        sinTheta * __sinf(phi),
        cosTheta);
}


__device__
inline float3 angleToBasis(float3 const& N, float3 const& N1, float3 const& N2, float const& phi, float const& theta)
{
    // angle relative to (N1, N2, N) basis is (sin(phi)cos(theta), sin(phi)sin(theta), cos(phi)).
    // perform a change of basis

    float3 base_vec = from_sphcoord(phi, theta);

    // representation of the matrix

    // return M*base_vec;
    // M is the matrix
    // \begin{pmatrix}
    // -\sin(\phi) & \cos(\theta)\cos(\phi) & \sin(\theta)\cos(\phi) \\
    // \cos(\phi) & \cos(\theta)\sin(\phi) & \sin(\theta)\sin(\phi) \\
    // 0 & -\sin(\theta) & \cos(\theta)
    // \end{pmatrix}

    // ( N1 N2 N) column
    return matrix3_multiply(N1, N2, N, base_vec);
}

__device__
inline void float3_addinplace(float3& target, float3 const& val, float const& scale)
{
    target.x += (val.x * scale);
    target.y += (val.y * scale);
    target.z += (val.z * scale);
}
