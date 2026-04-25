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
static void *vulkanHandle = nullptr;

bool tryLoadVulkan()
{
    if (vulkanHandle)
        return true;  // already loaded

    // Try common Vulkan loader library names for different platforms.
    const char *candidates[] = {
        "libvulkan.so.1",
        "libvulkan.so"
    };

    for (const char *name : candidates) {
        vulkanHandle = dlopen(name, RTLD_NOW | RTLD_GLOBAL);
        if (vulkanHandle) {
            // Resolve vkGetInstanceProcAddr and initialise the dispatch table.
            PFN_vkGetInstanceProcAddr getInstanceProcAddr =
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

            // Quick sanity check: can we enumerate instance extensions?
            try {
                auto props = vk::enumerateInstanceLayerProperties();
                (void)props;  // suppress unused-variable warning
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
