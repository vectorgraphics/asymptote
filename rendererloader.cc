/*****
 * rendererloader.cc
 * Probe for Vulkan availability at runtime via dlopen, falling back to
 * OpenGL when Vulkan is unavailable.
 *
 * The main asy binary has zero Vulkan link-time dependencies.
 * All Vulkan-specific code lives in libasyvulkan.so, loaded at runtime.
 *****/

#include "rendererloader.h"
#include "camperror.h"

#ifdef HAVE_RENDERER

// Definition of the runtime-determined 'vulkan' flag declared in common.h.
bool vulkan = false;

#if defined(HAVE_LIBGL)
#include "glrender.h"
#endif

#ifndef _WIN32
#include <dlfcn.h>
#include <unistd.h>
#endif
#include <iostream>
#include <pthread.h>
#include <cstring>
#include <string>
#include <vector>

#include "settings.h"    // for settings::verbose

#ifdef _WIN32
#include <windows.h>
#include <libloaderapi.h>
#endif

namespace camp {

static bool initializedRenderer = false;

// Opaque handle to the loaded Vulkan shared library.
#ifdef _WIN32
static HMODULE vulkanLibHandle = nullptr;
#else
static void *vulkanLibHandle = nullptr;
#endif

/**
 * Resolve the path to libasyvulkan.so relative to the executable directory.
 * On Linux we read /proc/self/exe to find our own directory.
 */
#ifndef _WIN32
static std::string resolveVulkanLibPath()
{
    // Strategy 1: Same directory as the executable (via /proc/self/exe).
    char exePath[4096];
    ssize_t len = readlink("/proc/self/exe", exePath, sizeof(exePath) - 1);
    if (len > 0) {
        exePath[len] = '\0';
        // Find last '/' to get directory.
        char *lastSlash = strrchr(exePath, '/');
        if (lastSlash) {
            *(lastSlash + 1) = '\0'; // Truncate after the slash to keep just the directory.
            std::string path = exePath;
            path += "libasyvulkan.so";
            return path;
        }
    }

    // Strategy 2: Current working directory.
    return "./libasyvulkan.so";
}
#endif

/**
 * Attempt to load libasyvulkan.so and create a Vulkan renderer.
 * Returns true on success (gl is set to the new AsyVkRender, vulkan=true).
 * Returns false on failure (gl remains as OpenGL or nullptr, vulkan=false).
 */
static bool tryLoadVulkanLib()
{
#ifdef _WIN32
    // On Windows, try LoadLibrary.
    vulkanLibHandle = LoadLibraryA("libasyvulkan.dll");
    if (!vulkanLibHandle) {
        if (settings::verbose > 1)
            std::cout << "Failed to load libasyvulkan.dll; falling back to OpenGL"
                      << std::endl;
        return false;
    }

    // Get the factory function.
    typedef void *(*CreateAsyVkRenderFn)();
    CreateAsyVkRenderFn fn =
        reinterpret_cast<CreateAsyVkRenderFn>(GetProcAddress(vulkanLibHandle, "createAsyVkRender"));
#else
    // On Unix, try multiple paths.
    std::vector<std::string> paths;

    // Path 1: Same directory as executable (via /proc/self/exe).
    paths.push_back(resolveVulkanLibPath());

    // Path 2: Just the library name (lets LD_LIBRARY_PATH / rpath do the work).
    paths.push_back("libasyvulkan.so");

    for (const auto &path : paths) {
        vulkanLibHandle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (vulkanLibHandle) {
            if (settings::verbose > 2)
                std::cout << "Loaded libasyvulkan.so from: " << path << std::endl;
            break;
        }
    }

    if (!vulkanLibHandle) {
        if (settings::verbose > 1)
            std::cout << "Failed to load libasyvulkan.so ("
                      << dlerror() << "); falling back to OpenGL" << std::endl;
        return false;
    }

    // Get the factory function.
    typedef void *(*CreateAsyVkRenderFn)();
    CreateAsyVkRenderFn fn =
        reinterpret_cast<CreateAsyVkRenderFn>(dlsym(vulkanLibHandle, "createAsyVkRender"));
#endif

    if (!fn) {
        if (settings::verbose > 1)
            std::cout << "Failed to find createAsyVkRender in libasyvulkan.so"
                      << std::endl;
        return false;
    }

    // Create the Vulkan renderer via the factory function.
    void *vkRenderer = fn();
    if (!vkRenderer) {
        if (settings::verbose > 1)
            std::cout << "createAsyVkRender() returned NULL" << std::endl;
        return false;
    }

    gl = static_cast<camp::AsyRender*>(vkRenderer);

#ifdef HAVE_PTHREAD
    if (gl)
        gl->mainthread = pthread_self();
#endif

    vulkan = true;
    return true;
}

/**
 * Lightweight probe: check if libasyvulkan.so can be loaded.
 * Used by settings.cc to report enabled/disabled options.
 * Does NOT create a renderer or set the global vulkan flag.
 */
bool tryLoadVulkan()
{
#ifdef _WIN32
    HMODULE h = LoadLibraryA("libasyvulkan.dll");
    if (h) {
        FreeLibrary(h);
        return true;
    }
    return false;
#else
    // Try the same paths as tryLoadVulkanLib but without keeping the handle.
    std::vector<std::string> paths;
    paths.push_back(resolveVulkanLibPath());
    paths.push_back("libasyvulkan.so");

    for (const auto &path : paths) {
        void *h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (h) {
            dlclose(h);
            return true;
        }
    }
    return false;
#endif
}

void unloadVulkan()
{
    if (vulkanLibHandle) {
#ifdef _WIN32
        FreeLibrary(vulkanLibHandle);
#else
        dlclose(vulkanLibHandle);
#endif
        vulkanLibHandle = nullptr;
    }
}

/**
 * Create the renderer object without performing any GPU/Vulkan probing.
 * Called from main.cc before starting threads so that gl is non-null and
 * the render thread can safely access gl->wait(...).
 *
 * Strategy: always create an OpenGL renderer first as a safe default.
 * If the user requested Vulkan (-vulkan, which is the default), attempt
 * to load libasyvulkan.so via dlopen and replace the renderer if successful.
 */
void createRenderer()
{
    if (gl != nullptr)
        return; // Already created

    bool useVulkan = settings::getSetting<bool>("vulkan");

    // Always create OpenGL renderer as the base/default.
#ifdef HAVE_LIBGL
    gl = new AsyGLRender();
#endif

    // If user wants Vulkan, try to load the shared library.
    if (useVulkan) {
        if (tryLoadVulkanLib()) {
            // Vulkan loaded successfully; gl is now AsyVkRender, vulkan=true.
            if (settings::verbose > 1)
                std::cout << "Using Vulkan renderer" << std::endl;
            return;
        }
        // Vulkan failed to load; keep OpenGL renderer, vulkan stays false.
        if (settings::verbose > 1)
            std::cout << "Using OpenGL renderer" << std::endl;
    } else {
        // User explicitly requested no Vulkan (-novulkan).
        if (settings::verbose > 1)
            std::cout << "Using OpenGL renderer" << std::endl;
    }

    vulkan = false;

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

    // The vulkan flag is already set correctly by createRenderer().
    // Just log at verbose level 3+ for confirmation.
    if (settings::verbose > 2) {
        if (vulkan)
            std::cout << "Using Vulkan renderer" << std::endl;
        else
            std::cout << "Using OpenGL renderer" << std::endl;
    }
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
