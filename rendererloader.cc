/*****
 * rendererloader.cc
 * Dynamically load the Vulkan loader library at runtime, falling back
 * to OpenGL when Vulkan is unavailable.
 *****/

#include "rendererloader.h"

#ifdef HAVE_RENDERER

// Definition of the runtime-determined 'vulkan' flag declared in common.h.
bool vulkan = false;

#if defined(HAVE_LIBVULKAN)

#include <dlfcn.h>
#include <iostream>

#include "vk.h"          // pulls in vulkan.hpp with dynamic dispatch
#include "settings.h"    // for settings::verbose

namespace camp {

// Holds the dlopen handle for libvulkan.so (or equivalent).
// Null when Vulkan was linked statically at build time.
static void *vulkanHandle = nullptr;

bool tryLoadVulkan()
{
    if (vulkanHandle)
        return true;  // already loaded via dlopen

    // When -lvulkan is linked at build time, vkGetInstanceProcAddr is
    // already resolved by the dynamic linker.  We can probe Vulkan
    // availability directly without dlopen.
    PFN_vkGetInstanceProcAddr getInstanceProcAddr =
        reinterpret_cast<PFN_vkGetInstanceProcAddr>(
            dlsym(RTLD_DEFAULT, "vkGetInstanceProcAddr"));

    if (getInstanceProcAddr) {
        // The Vulkan loader is already available (linked or pre-loaded).
        VULKAN_HPP_DEFAULT_DISPATCHER.init(getInstanceProcAddr);

        // Quick sanity check: can we enumerate instance layers?
        try {
            auto props = vk::enumerateInstanceLayerProperties();
            (void)props;  // suppress unused-variable warning
        } catch (const std::exception &e) {
            if (settings::verbose > 1)
                std::cerr << "warning: Vulkan instance enumeration failed ("
                          << e.what() << ")" << std::endl;
            return false;
        }

        if (settings::verbose > 1)
            std::cout << "Vulkan available (linked at build time)" << std::endl;
        return true;
    }

    // Not linked — try dlopen as a last resort.
    const char *candidates[] = {
        "libvulkan.so.1",
        "libvulkan.so"
    };

    for (const char *name : candidates) {
        vulkanHandle = dlopen(name, RTLD_NOW | RTLD_GLOBAL);
        if (vulkanHandle) {
            getInstanceProcAddr =
                reinterpret_cast<PFN_vkGetInstanceProcAddr>(
                    dlsym(vulkanHandle, "vkGetInstanceProcAddr"));

            if (!getInstanceProcAddr) {
                std::cerr << "warning: could not resolve vkGetInstanceProcAddr in "
                          << name << std::endl;
                dlclose(vulkanHandle);
                vulkanHandle = nullptr;
                continue;
            }

            VULKAN_HPP_DEFAULT_DISPATCHER.init(getInstanceProcAddr);

            try {
                auto props = vk::enumerateInstanceLayerProperties();
                (void)props;
            } catch (const std::exception &e) {
                std::cerr << "warning: Vulkan instance enumeration failed ("
                          << e.what() << ")" << std::endl;
                dlclose(vulkanHandle);
                vulkanHandle = nullptr;
                continue;
            }

            if (settings::verbose > 1)
                std::cout << "Vulkan loaded successfully from " << name
                          << std::endl;
            return true;
        }
    }

    if (settings::verbose > 1)
        std::cout << "Vulkan loader library not found; falling back to OpenGL"
                  << std::endl;
    return false;
}

void unloadVulkan()
{
    if (vulkanHandle) {
        dlclose(vulkanHandle);
        vulkanHandle = nullptr;
    }
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
