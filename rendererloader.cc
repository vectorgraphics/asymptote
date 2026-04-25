/*****
 * rendererloader.cc
 * Probe for Vulkan availability at runtime, falling back to OpenGL
 * when Vulkan is unavailable.
 *****/

#include "rendererloader.h"

#ifdef HAVE_RENDERER

// Definition of the runtime-determined 'vulkan' flag declared in common.h.
bool vulkan = false;

#if defined(HAVE_LIBVULKAN)

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <iostream>

#include "vk.h"          // pulls in vulkan.hpp with dynamic dispatch
#include "settings.h"    // for settings::verbose

namespace camp {

bool tryLoadVulkan()
{
    // Resolve vkGetInstanceProcAddr from the process address space.
    // This works whether Vulkan was linked at build time or loaded
    // by a dependent library (e.g., vcpkg's vulkan-loader).
#if defined(_WIN32)
    PFN_vkGetInstanceProcAddr getInstanceProcAddr =
        reinterpret_cast<PFN_vkGetInstanceProcAddr>(
            GetProcAddress(GetModuleHandleA(nullptr), "vkGetInstanceProcAddr"));
#else
    PFN_vkGetInstanceProcAddr getInstanceProcAddr =
        reinterpret_cast<PFN_vkGetInstanceProcAddr>(
            dlsym(RTLD_DEFAULT, "vkGetInstanceProcAddr"));
#endif

    if (!getInstanceProcAddr) {
        if (settings::verbose > 1)
            std::cout << "Vulkan loader not found; falling back to OpenGL"
                      << std::endl;
        return false;
    }

    VULKAN_HPP_DEFAULT_DISPATCHER.init(getInstanceProcAddr);

    // Sanity check: can we enumerate instance layers?
    try {
        auto props = vk::enumerateInstanceLayerProperties();
        (void)props;
    } catch (const std::exception &e) {
        if (settings::verbose > 1)
            std::cerr << "warning: Vulkan instance enumeration failed ("
                      << e.what() << ")" << std::endl;
        return false;
    }

    if (settings::verbose > 1)
        std::cout << "Vulkan available" << std::endl;
    return true;
}

void unloadVulkan()
{
    // Nothing to unload -- Vulkan is linked, not dynamically loaded.
}

} // namespace camp

#else // !HAVE_LIBVULKAN

// No Vulkan support compiled in at all.
namespace camp {

bool tryLoadVulkan() { return false; }
void unloadVulkan() {}

} // namespace camp

#endif // HAVE_LIBVULKAN

#else // !HAVE_RENDERER

namespace camp {

bool tryLoadVulkan() { return false; }
void unloadVulkan() {}

} // namespace camp

#endif // HAVE_RENDERER
