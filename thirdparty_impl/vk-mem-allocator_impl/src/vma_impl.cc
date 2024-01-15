/**
 * @file vma_impl.cc
 * @author Supakorn "Jamie" Rassameemasmuang (jamievlin [at] outlook.com)
 * @brief Implementation file for VulkanMemoryAllocator
 */

#define VMA_IMPLEMENTATION

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define VK_USE_PLATFORM_WIN32_KHR
#endif

#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1

// ReSharper disable once CppUnusedIncludeDirective
#include <vk_mem_alloc.h>
