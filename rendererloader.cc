/*****
 * rendererloader.cc
 * Probe for Vulkan availability at runtime, falling back to OpenGL
 * when Vulkan is unavailable.
 *****/

#include "rendererloader.h"
#include "camperror.h"

#ifdef HAVE_RENDERER

// Definition of the runtime-determined 'vulkan' flag declared in common.h.
bool vulkan = false;

#if defined(HAVE_LIBVULKAN)
#include "vkrender.h"
#endif
#if defined(HAVE_LIBGL)
#include "glrender.h"
#endif

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <iostream>
#include <pthread.h>

#include "vk.h"          // pulls in vulkan.hpp with dynamic dispatch
#include "settings.h"    // for settings::verbose

namespace camp {

static bool initializedRenderer = false;

bool tryLoadVulkan()
{
#if defined(HAVE_LIBVULKAN)
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
#else
    return false;
#endif // HAVE_LIBVULKAN
}

void unloadVulkan()
{
    // Nothing to unload -- Vulkan is linked, not dynamically loaded.
}

/**
 * Create the renderer object without performing any GPU/Vulkan probing.
 * Called from main.cc before starting threads so that gl is non-null and
 * the render thread can safely access gl->wait(...).
 * The constructor of AsyVkRender/AsyGLRender is trivial (= default);
 * no actual graphics-library initialisation occurs here.
 *
 * Type selection: respect the user's -vulkan/-novulkan setting combined with
 * compile-time availability.  No runtime Vulkan probe is performed here.
 */
void createRenderer()
{
    if (gl != nullptr)
        return; // Already created

    bool useVulkan = settings::getSetting<bool>("vulkan");

#ifdef HAVE_LIBVULKAN
    if (useVulkan) {
        gl = new AsyVkRender();
    } else
#endif
#ifdef HAVE_LIBGL
    {
        gl = new AsyGLRender();
    }
#endif

    if (!gl) {
        camp::reportError("No 3D rendering library available");
    }

#ifdef HAVE_PTHREAD
    if (gl)
        gl->mainthread = pthread_self();
#endif
}

void initRenderer()
{
    if (initializedRenderer)
        return; // Already fully initialised

    initializedRenderer = true;

    bool useVulkan = settings::getSetting<bool>("vulkan");
    vulkan = useVulkan && tryLoadVulkan();

#ifdef HAVE_LIBGL
    if (!vulkan && settings::verbose > 1)
        std::cout << "Using OpenGL renderer" << std::endl;
#endif
}

} // namespace camp

#else // !HAVE_RENDERER

namespace camp {

static bool initializedRenderer = false;

bool tryLoadVulkan() { return false; }
void unloadVulkan() {}
void createRenderer() {}
void initRenderer() {}

} // namespace camp

#endif // HAVE_RENDERER
