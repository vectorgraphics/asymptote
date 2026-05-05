/**
 * @file vulkanshim.cc
 * C-exported entry point for the Vulkan renderer shared library.
 * This is the only file that needs to be compiled into libasyvulkan.so
 * alongside vkrender.o, vkutils.o, vkdispatchstorage.o, vma_cxx.o, vma_impl.o.
 */

#include <iostream>
#include <stdexcept>

#include "common.h"

#ifdef HAVE_LIBGLM
#include "vk.h"
#include "vkrender.h"
#endif

extern "C" {

/**
 * Create and return a new AsyVkRender instance.
 * The caller (rendererloader.cc) receives this as a void* and casts it
 * back to the appropriate type.  Returns NULL on failure.
 */
void *createAsyVkRender()
{
#ifdef HAVE_LIBVULKAN
#ifdef HAVE_LIBGLM
    try {
        // Initialize the Vulkan-Hpp dynamic dispatcher with global functions.
        // Since libasyvulkan.so links against -lvulkan, vkGetInstanceProcAddr
        // is resolved directly by the dynamic linker.
        VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

        return new camp::AsyVkRender();
    } catch (const std::exception &e) {
        std::cerr << "createAsyVkRender exception: " << e.what() << std::endl;
        return nullptr;
    } catch (...) {
        std::cerr << "createAsyVkRender: unknown exception" << std::endl;
        return nullptr;
    }
#else
    return nullptr;
#endif
#else
    return nullptr;
#endif
}

} // extern "C"
