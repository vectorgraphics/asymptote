/**
 * @file vk.h
 * @author Supakorn "Jamie" Rassameemasmuang (jamievlin [at] outlook.com)
 * @brief Common include header for vulkan
 */

#pragma once

#if defined(HAVE_VULKAN)

// We undefined NDEBUG for common.h, but some files
// do not use common.h, causing includes
// to be a mix of NDEBUG-ed vulkan header and those without
#undef NDEBUG
#define VK_ENABLE_BETA_EXTENSIONS

#if defined(_WIN32)
#define VK_USE_PLATFORM_WIN32_KHR
#endif

#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>

#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <vk_mem_alloc.h>

#endif
