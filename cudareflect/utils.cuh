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
inline glm::vec2 to_sphcoord(glm::vec3 const& vec)
{
    return glm::vec2(
        atan2f(-vec.y, -vec.x) + PI,
        acosf(vec.z)
    );
}

__device__
inline glm::vec3 from_sphcoord(float const& phi, float const& theta)
{
    return glm::vec3(
        __sinf(theta) * __cosf(phi),
        __sinf(theta) * __sinf(phi),
        __cosf(theta));
}

__device__
inline glm::vec3 from_sphcoord(float const& phi, float const& cosTheta, float const& sinTheta)
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

    glm::vec3 base_vec = from_sphcoord(phi, theta);

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

struct Vec3Utility
{
    __device__ static glm::vec3 init()
    {
        return glm::vec3(0.0f);
    }

    __device__ static float abs2(glm::vec3 v)
    {
        return v.x * v.x + v.y * v.y + v.z * v.z;
    }
};
