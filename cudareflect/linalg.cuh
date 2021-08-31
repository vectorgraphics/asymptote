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
inline float3 from_sphcoord(float const& phi, float const& theta)
{
    return make_float3(
        __sinf(theta) * __cosf(phi),
        __sinf(theta) * __sinf(phi),
        __cosf(theta));
}

__device__
inline glm::vec2 to_sphcoord(glm::vec3 const& vec)
{
    return glm::vec2(
        atan2f(-vec.y, -vec.x) + PI,
        acosf(vec.z)
    );
}

__device__
inline glm::vec3 from_sphcoord_glm(float const& phi, float const& theta)
{
    return glm::vec3(
        __sinf(theta) * __cosf(phi),
        __sinf(theta) * __sinf(phi),
        __cosf(theta));
}

__device__
inline glm::vec3 from_sphcoord_glm(float const& phi, float const& cosTheta, float const& sinTheta)
{
    return glm::vec3(
        sinTheta * __cosf(phi),
        sinTheta * __sinf(phi),
        cosTheta);
}

__device__
inline glm::vec3 angleToBasis(glm::mat3 const& normalOrthBasis, float const& phi, float const& theta)
{
    // angle relative to (N1, N2, N) basis is (sin(phi)cos(theta), sin(phi)sin(theta), cos(phi)).
    // perform a change of basis

    glm::vec3 base_vec = from_sphcoord_glm(phi, theta);

    // representation of the matrix

    // return M*base_vec;
    // M is the matrix
    // \begin{pmatrix}
    // -\sin(\phi) & \cos(\theta)\cos(\phi) & \sin(\theta)\cos(\phi) \\
    // \cos(\phi) & \cos(\theta)\sin(\phi) & \sin(\theta)\sin(\phi) \\
    // 0 & -\sin(\theta) & \cos(\theta)
    // \end{pmatrix}

    // ( N1 N2 N) column
    return normalOrthBasis * base_vec;
}

__device__
inline void float3_addinplace(float3& target, float3 const& val, float const& scale)
{
    target.x += (val.x * scale);
    target.y += (val.y * scale);
    target.z += (val.z * scale);
}

struct DefaultVec3ZeroInit
{
    __device__ static glm::vec3 init()
    {
        return glm::vec3(0.0f);
    }
};

template<typename TRet=glm::vec3, typename TInit=DefaultVec3ZeroInit, typename T>
__device__ inline TRet simpsonThird(T fn, float const& a, float const& b, size_t const& Nhalf)
{
    float const NVal = 2.0f * Nhalf;
    float const Hvalue = (b - a) / NVal;

    TRet base = fn(a);

    TRet sumeven = TInit::init();
    for (size_t i = 2; i < NVal; i += 2)
    {
        sumeven +=  fn(a + (i * Hvalue));
    }
    base += (2.0f * sumeven);


    TRet sumodd = TInit::init();
    for (size_t i = 2; i <= NVal; i += 2)
    {
        sumodd += fn(a + ((i - 1) * Hvalue));
    }

    base += (4.0f * sumodd);
    base += fn(b);
    return base * (Hvalue / 3.0f);
}