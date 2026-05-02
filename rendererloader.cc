/*****
 * rendererloader.cc
 * On Unix: Probe for Vulkan and OpenGL availability at runtime via dlopen.
 * On Windows: Vulkan is statically linked; the renderer is directly instantiated.
 *
 * The main asy binary has zero link-time dependencies on Vulkan or OpenGL
 * on Unix. All Vulkan-specific code lives in libasyvulkan.so, loaded at runtime.
 * All OpenGL-specific code lives in libasyopengl.so, loaded at runtime.
 *
 * On Windows, Vulkan is linked directly into the asy binary.
 *
 * For WebGL (html) and v3d output, AsyWebGLRender is used instead, which
 * requires no GPU libraries - it only sets up state for client-side rendering.
 *****/

#include "rendererloader.h"
#include "camperror.h"
#include "renderBase.h"
#include "webglrender.h"

#ifdef _WIN32
#include "vkrender.h"
#endif

#ifdef HAVE_RENDERER

// Definition of the runtime-determined 'vulkan' flag declared in common.h.
bool vulkan = false;

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

// Opaque handles to the loaded renderer shared libraries.
#ifdef _WIN32
static HMODULE vulkanLibHandle = nullptr;
static HMODULE glLibHandle = nullptr;
#else
static void *vulkanLibHandle = nullptr;
static void *glLibHandle = nullptr;
#endif

/**
 * Resolve the path to a renderer shared library relative to the executable directory.
 * On Linux we read /proc/self/exe to find our own directory.
 */
#ifndef _WIN32
static std::string resolveRendererLibPath(const std::string &libName)
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
            path += libName;
            return path;
        }
    }

    // Strategy 2: Current working directory.
    return "./" + libName;
}
#endif

/**
 * Attempt to load libasyvulkan.so and create a Vulkan renderer.
 * Returns true on success (gl is set to the new AsyVkRender, vulkan=true).
 * Returns false on failure (gl remains nullptr, vulkan=false).
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
    paths.push_back(resolveRendererLibPath("libasyvulkan.so"));

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
 * Attempt to load libasyopengl.so and create an OpenGL renderer.
 * Returns true on success (gl is set to the new AsyGLRender, vulkan=false).
 * Returns false on failure (gl remains nullptr).
 */
static bool tryLoadOpenGLLib()
{
#ifdef _WIN32
    // On Windows, try LoadLibrary.
    glLibHandle = LoadLibraryA("libasyopengl.dll");
    if (!glLibHandle) {
        if (settings::verbose > 1)
            std::cout << "Failed to load libasyopengl.dll" << std::endl;
        return false;
    }

    // Get the factory function.
    typedef void *(*CreateAsyGLRenderFn)();
    CreateAsyGLRenderFn fn =
        reinterpret_cast<CreateAsyGLRenderFn>(GetProcAddress(glLibHandle, "createAsyGLRender"));
#else
    // On Unix, try multiple paths.
    std::vector<std::string> paths;

    // Path 1: Same directory as executable (via /proc/self/exe).
    paths.push_back(resolveRendererLibPath("libasyopengl.so"));

    // Path 2: Just the library name (lets LD_LIBRARY_PATH / rpath do the work).
    paths.push_back("libasyopengl.so");

    for (const auto &path : paths) {
        glLibHandle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (glLibHandle) {
            if (settings::verbose > 2)
                std::cout << "Loaded libasyopengl.so from: " << path << std::endl;
            break;
        }
    }

    if (!glLibHandle) {
        if (settings::verbose > 1)
            std::cout << "Failed to load libasyopengl.so ("
                      << dlerror() << ")" << std::endl;
        return false;
    }

    // Get the factory function.
    typedef void *(*CreateAsyGLRenderFn)();
    CreateAsyGLRenderFn fn =
        reinterpret_cast<CreateAsyGLRenderFn>(dlsym(glLibHandle, "createAsyGLRender"));
#endif

    if (!fn) {
        if (settings::verbose > 1)
            std::cout << "Failed to find createAsyGLRender in libasyopengl.so"
                      << std::endl;
        return false;
    }

    // Create the OpenGL renderer via the factory function.
    void *glRenderer = fn();
    if (!glRenderer) {
        if (settings::verbose > 1)
            std::cout << "createAsyGLRender() returned NULL" << std::endl;
        return false;
    }

    gl = static_cast<camp::AsyRender*>(glRenderer);

#ifdef HAVE_PTHREAD
    if (gl)
        gl->mainthread = pthread_self();
#endif

    vulkan = false;
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
    // On Windows, Vulkan is statically linked; always available.
    return true;
#else
    // Try the same paths as tryLoadVulkanLib but without keeping the handle.
    std::vector<std::string> paths;
    paths.push_back(resolveRendererLibPath("libasyvulkan.so"));
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

/**
 * Lightweight probe: check if libasyopengl.so can be loaded.
 * Used by settings.cc to report enabled/disabled options.
 * Does NOT create a renderer or set the global vulkan flag.
 */
bool tryLoadOpenGL()
{
#ifdef _WIN32
    // On Windows, OpenGL is not supported; only Vulkan is available.
    return false;
#else
    std::vector<std::string> paths;
    paths.push_back(resolveRendererLibPath("libasyopengl.so"));
    paths.push_back("libasyopengl.so");

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

void unloadOpenGL()
{
    if (glLibHandle) {
#ifdef _WIN32
        FreeLibrary(glLibHandle);
#else
        dlclose(glLibHandle);
#endif
        glLibHandle = nullptr;
    }
}

/**
 * Create a WebGL renderer for html/v3d output.
 * This does NOT require Vulkan or OpenGL libraries - it only sets up state
 * needed by jsfile.cc and v3dfile.cc to generate the output files.
 *
 * If a Vulkan/OpenGL renderer was already created (e.g., by createRenderer()),
 * we replace the global pointer with AsyWebGLRender. The old renderer is NOT
 * deleted to avoid triggering cleanup code (like glslang::FinalizeProcess())
 * that can cause issues when called during program shutdown.
 */
static void createWebGLRenderer()
{
    // Replace the global pointer without deleting the old renderer.
    // This avoids triggering Vulkan/OpenGL cleanup code that can cause
    // assertion failures at program exit (e.g., glslang::FinalizeProcess()).
    // The old renderer will be cleaned up when the process exits.
    if (gl != nullptr) {
        // Don't delete - just leak the old renderer (acceptable for short-lived processes)
        gl = nullptr;
    }

    gl = new camp::AsyWebGLRender();

#ifdef HAVE_PTHREAD
    if (gl)
        gl->mainthread = pthread_self();
#endif

    vulkan = false;  // WebGL renderer is not Vulkan
}

/**
 * Create the renderer object without performing any GPU/Vulkan probing.
 * Called from main.cc before starting threads so that gl is non-null and
 * the render thread can safely access gl->wait(...).
 *
 * On Windows: Vulkan is statically linked; directly instantiate AsyVkRender.
 * On Unix: Try to load the requested renderer via dlopen/LoadLibrary.
 * If the user requested Vulkan (-vulkan, which is the default), attempt
 * to load libasyvulkan.so first. If that fails or the user requested
 * OpenGL (-novulkan), load libasyopengl.so as a fallback.
 */
void createRenderer()
{
    if (gl != nullptr)
        return; // Already created

    bool useVulkan = settings::getSetting<bool>("vulkan");

#ifdef _WIN32
    // On Windows, Vulkan is statically linked into the binary.
    // Initialize the Vulkan-Hpp dynamic dispatcher before creating the renderer,
    // matching what createAsyVkRender() in vulkanshim.cc does on Unix.
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

    // Directly instantiate the Vulkan renderer (no OpenGL fallback available).
    gl = new camp::AsyVkRender();
#ifdef HAVE_PTHREAD
    if (gl)
        gl->mainthread = pthread_self();
#endif
    vulkan = true;
    if (settings::verbose > 1)
        std::cout << "Using Vulkan renderer" << std::endl;
#else
    // If user wants Vulkan, try to load the shared library first.
    if (useVulkan) {
        if (tryLoadVulkanLib()) {
            // Vulkan loaded successfully; gl is now AsyVkRender, vulkan=true.
            if (settings::verbose > 1)
                std::cout << "Using Vulkan renderer" << std::endl;
            return;
        }
        // Vulkan failed to load; fall through to try OpenGL.
        if (settings::verbose > 1)
            std::cout << "Vulkan unavailable, falling back to OpenGL" << std::endl;
    }

    // Try to load the OpenGL renderer.
    if (tryLoadOpenGLLib()) {
        if (settings::verbose > 1)
            std::cout << "Using OpenGL renderer" << std::endl;
        return;
    }
#endif // _WIN32

    // Both renderers failed to load. Leave gl as nullptr; the error will be
    // reported lazily in initRenderer() when 3D rendering is actually requested.
    vulkan = false;
}

/**
 * Initialise the renderer, optionally selecting based on output format.
 *
 * For WebGL (html) and v3d formats, creates AsyWebGLRender which requires
 * no GPU libraries - it only sets up state for client-side rendering.
 *
 * @param format Output format string (e.g., "html", "v3d", or empty/NULL for default)
 */
void initRenderer(const char* format)
{
    // For WebGL and v3d output, use the lightweight AsyWebGLRender
    // which doesn't require Vulkan or OpenGL libraries
    bool isFormat3D = (format != nullptr &&
                       (strcmp(format, "html") == 0 || strcmp(format, "v3d") == 0));

    // If we have a WebGL renderer but now need GPU rendering (or vice versa),
    // reset and re-initialize with the appropriate renderer
    if (initializedRenderer) {
        bool currentIsWebGL = dynamic_cast<AsyWebGLRender*>(gl) != nullptr;
        if (currentIsWebGL != isFormat3D) {
            initializedRenderer = false;
            // Clear the old renderer so a new one gets created below.
            // We don't delete it to avoid triggering cleanup code
            // (e.g., glslang::FinalizeProcess()) that can cause issues.
            if (currentIsWebGL)
                gl = nullptr;
        }
    }

    if (initializedRenderer)
        return; // Already fully initialised for this format type

    if (isFormat3D) {
        createWebGLRenderer();
    }

    if (gl == nullptr) {
        // For non-format3d output, try Vulkan/OpenGL renderers
        camp::reportError("No 3D rendering library available");
    }

    initializedRenderer = true;

    if (settings::verbose > 2) {
        if (isFormat3D && format)
            std::cout << "Using WebGL renderer for " << format << " output" << std::endl;
        else if (vulkan)
            std::cout << "Using Vulkan renderer" << std::endl;
        else
            std::cout << "Using OpenGL renderer" << std::endl;
    }
}

} // namespace camp

#else // !HAVE_RENDERER

namespace camp {

bool tryLoadVulkan() { return false; }
bool tryLoadOpenGL() { return false; }
void unloadVulkan() {}
void unloadOpenGL() {}
void createRenderer() {}
void initRenderer(const char*) {}

} // namespace camp

#endif // HAVE_RENDERER
