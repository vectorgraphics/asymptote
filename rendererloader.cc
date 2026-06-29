/*****
 * rendererloader.cc
 * On Unix: Probe for Vulkan and OpenGL availability at runtime via dlopen.
 * On Windows: Vulkan is linked against vulkan-1.dll; the renderer is
 *   directly instantiated.  If no hardware GPU is detected (e.g., pre-2012
 *   hardware, VirtualBox VM), a llvmpipe fallback is activated by writing
 *   an ICD manifest (lvp_icd.json) and loading vulkan_lvp.dll (Lavapipe).
 * On macOS (Intel x86_64): If Metal is unavailable (e.g., headless VM),
 *   fall back to a shipped llvmpipe driver by locating libvulkan_lvp.dylib
 *   via the Asymptote search path and writing an ICD manifest.
 *   Apple Silicon (arm64) is not supported by llvmpipe; use SwiftShader instead.
 *
 * The main asy binary has zero link-time dependencies on Vulkan or OpenGL
 * on Unix. All Vulkan-specific code lives in libasyvulkan.so, loaded at runtime.
 * All OpenGL-specific code lives in libasyopengl.so, loaded at runtime.
 *
 * On Windows, Vulkan is linked directly into the asy binary.
 *
 * For WebGL (html) and v3d output, NoRender is used instead, which
 * requires no GPU libraries - it only sets up state for client-side rendering.
 *****/

#include "rendererloader.h"
#include "camperror.h"

#ifdef __APPLE__
// Include glfw3.h with Vulkan support BEFORE renderBase.h, which includes
// it with GLFW_INCLUDE_NONE.  This ensures we get both the Vulkan types
// (PFN_vkGetInstanceProcAddr) and the glfwInitVulkanLoader declaration.
#define VK_NO_PROTOTYPES
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#endif

#include "renderBase.h"
#include "norender.h"

// Defined here (in asycore) rather than in main.cc so that executables which
// link asycore without main.cc (e.g. asyCxxTests) still resolve these symbols,
// which are referenced from picture.cc and below.
pthread_mutex_t main_wait_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t main_wait_cond = PTHREAD_COND_INITIALIZER;

#ifdef _WIN32
#include "vkrender.h"
#endif

bool vulkan = false;

#ifdef HAVE_RENDERER

#ifdef _WIN32
#include <windows.h>
#include <libloaderapi.h>
// For llvmpipe fallback: we need raw Vulkan types to probe device availability.
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#else
#include <dlfcn.h>
#endif

#include <cstring>
#include <iostream>
#include <pthread.h>
#include <string>

#include "settings.h"    // for settings::verbose
#include "locate.h"       // for settings::locateFile

namespace camp {

bool headlessRenderer = false;  // True when using software renderer without display (e.g., llvmpipe on macOS without Metal)
static bool initializedRenderer = false;

static void signalRendererReady()
{
#ifdef HAVE_PTHREAD
    pthread_mutex_lock(&main_wait_mutex);
    pthread_cond_broadcast(&main_wait_cond);
    pthread_mutex_unlock(&main_wait_mutex);
#endif
}

// Opaque handles to the loaded renderer shared libraries.
#ifdef _WIN32
static HMODULE vulkanLibHandle = nullptr;
static HMODULE glLibHandle = nullptr;
static HMODULE lvpLibHandle = nullptr;
#else
static void *vulkanLibHandle = nullptr;
static void *glLibHandle = nullptr;
#endif

/**
 * Load a renderer shared library by searching the Asymptote path.
 * Returns a valid handle on success, or nullptr on failure.
 */
#ifndef _WIN32
static void *loadRendererLib(const char *libName)
{
    mem::string locPath = settings::locateFile(libName, true, "");
    std::string pathStr = mem::stdString(locPath);

    return dlopen(pathStr.c_str(), RTLD_NOW | RTLD_LOCAL);
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
    mem::string locPath = settings::locateFile("libasyvulkan.so", true, "");
    std::string pathStr = mem::stdString(locPath);

    vulkanLibHandle = dlopen(pathStr.c_str(), RTLD_NOW | RTLD_LOCAL);

    if (!vulkanLibHandle) {
        if (settings::verbose > 1)
            std::cout << "Failed to load libasyvulkan.so ("
                      << dlerror() << "); falling back to OpenGL" << std::endl;
        return false;
    }

    if (settings::verbose > 2)
        std::cout << "Loaded libasyvulkan.so from: " << pathStr << std::endl;

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
        gl->threadMgr.mainthread = pthread_self();
#endif

    signalRendererReady();
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
    mem::string locPath = settings::locateFile("libasyopengl.so", true, "");
    std::string pathStr = mem::stdString(locPath);

    glLibHandle = dlopen(pathStr.c_str(), RTLD_NOW | RTLD_LOCAL);

    if (!glLibHandle) {
        if (settings::verbose > 1)
            std::cout << "Failed to load libasyopengl.so ("
                      << dlerror() << ")" << std::endl;
        return false;
    }

    if (settings::verbose > 2)
        std::cout << "Loaded libasyopengl.so from: " << pathStr << std::endl;

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
        gl->threadMgr.mainthread = pthread_self();
#endif

    signalRendererReady();
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
    void *h = loadRendererLib("libasyvulkan.so");
    if (h) {
        dlclose(h);
        return true;
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
    void *h = loadRendererLib("libasyopengl.so");
    if (h) {
        dlclose(h);
        return true;
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

    // Also unload the llvmpipe fallback DLL on Windows.
#ifdef _WIN32
    if (lvpLibHandle) {
        FreeLibrary(lvpLibHandle);
        lvpLibHandle = nullptr;
    }
#endif
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
 * Create a NoRender instance for html/v3d output.
 * This does NOT require Vulkan or OpenGL libraries - it only sets up state
 * needed by jsfile.cc and v3dfile.cc to generate the output files.
 *
 * If a Vulkan/OpenGL renderer was already created (e.g., by createRenderer()),
 * we replace the global pointer with NoRender. The old renderer is NOT
 * deleted to avoid triggering cleanup code (like glslang::FinalizeProcess())
 * that can cause issues when called during program shutdown.
 */
static void createNoRenderer()
{
    // Replace the global pointer without deleting the old renderer.
    // This avoids triggering Vulkan/OpenGL cleanup code that can cause
    // assertion failures at program exit (e.g., glslang::FinalizeProcess()).
    // The old renderer will be cleaned up when the process exits.
    if (gl != nullptr) {
        // Don't delete - just leak the old renderer (acceptable for short-lived processes)
        gl = nullptr;
    }

    gl = new camp::NoRender();

#ifdef HAVE_PTHREAD
    if (gl)
        gl->threadMgr.mainthread = pthread_self();
#endif

    signalRendererReady();
#ifdef HAVE_RENDERER
    vulkan = false;  // NoRender is not Vulkan
#endif
}

/**
 * Create the renderer object by probing for Vulkan/OpenGL availability.
 * Called lazily from initRenderer() when shipout3() first needs GPU rendering.
 * On Unix this loads the appropriate shared library (libasyvulkan.so or
 * libasyopengl.so) via dlopen.  The render thread's glrenderWrapper() is
 * safe with a null gl pointer, so there is no need to pre-initialise the
 * renderer in main().
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
    // On Windows, Vulkan is linked against vulkan-1.dll.  Initialize the
    // Vulkan-Hpp dynamic dispatcher before creating the renderer.
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

    // Probe for a hardware GPU.  If none is available (e.g., pre-2012
    // hardware, VirtualBox VM), fall back to llvmpipe via Lavapipe.
    bool hasHardwareGPU = false;
    try {
        VkInstanceCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;

        VkInstance tmpInst;
        if (vkCreateInstance(&info, nullptr, &tmpInst) == VK_SUCCESS) {
            uint32_t devCount = 0;
            vkEnumeratePhysicalDevices(tmpInst, &devCount, nullptr);
            if (devCount > 0) {
                std::vector<VkPhysicalDevice> devices(devCount);
                vkEnumeratePhysicalDevices(tmpInst, &devCount, devices.data());
                for (uint32_t i = 0; i < devCount; ++i) {
                    VkPhysicalDeviceProperties props;
                    vkGetPhysicalDeviceProperties(devices[i], &props);
                    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ||
                        props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ||
                        props.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU) {
                        hasHardwareGPU = true;
                        break;
                    }
                }
            }
            vkDestroyInstance(tmpInst, nullptr);
        }
    } catch (...) {
        // If probing fails for any reason, proceed without the check.
    }

    if (!hasHardwareGPU) {
        // No hardware GPU found -- set up llvmpipe (Lavapipe) fallback.
        // Strategy: write lvp_icd.json next to the executable and set
        // VK_ICD_FILENAMES so the Vulkan loader picks it up on the next
        // instance creation.

        // 1) Determine the directory of our own executable.
        std::string exeDir;
        {
            char buf[MAX_PATH];
            DWORD len = GetModuleFileNameA(nullptr, buf, MAX_PATH);
            if (len > 0 && len < MAX_PATH) {
                std::string exePath(buf);
                size_t slash = exePath.find_last_of("\\/");
                if (slash != std::string::npos)
                    exeDir = exePath.substr(0, slash + 1);
            }
        }

        // 2) Write lvp_icd.json next to the executable.
        std::string icdName = "lvp_icd.json";
        std::string icdPath;
        if (!exeDir.empty())
            icdPath = exeDir + icdName;
        else
            icdPath = icdName;

        {
            std::ofstream ofs(icdPath.c_str(), std::ios::trunc);
            if (ofs.is_open()) {
                ofs << "{\n"
                    << "    \"file_format_version\": \"1.0.0\",\n"
                    << "    \"ICD\": {\n"
                    << "        \"library_path\": \".\\\\vulkan_lvp.dll\",\n"
                    << "        \"api_version\": \"1.3.0\"\n"
                    << "    }\n"
                    << "}\n";
                if (settings::verbose > 1)
                    std::cout << "Wrote Lavapipe ICD manifest: " << icdPath
                              << std::endl;
            }
        }

        // 3) Set VK_ICD_FILENAMES so the loader finds our manifest.
        _putenv_s("VK_ICD_FILENAMES", icdPath.c_str());

        // 4) Load vulkan_lvp.dll (the Lavapipe driver).
        std::string lvpDllPath;
        if (!exeDir.empty())
            lvpDllPath = exeDir + "vulkan_lvp.dll";
        else
            lvpDllPath = "vulkan_lvp.dll";

        lvpLibHandle = LoadLibraryA(lvpDllPath.c_str());
        if (lvpLibHandle) {
            if (settings::verbose > 1)
                std::cout << "Loaded llvmpipe fallback: " << lvpDllPath
                          << std::endl;
        } else {
            if (settings::verbose > 1)
                std::cout << "Warning: failed to load " << lvpDllPath
                          << "; proceeding without llvmpipe" << std::endl;
        }

        // 5) Re-initialize the Vulkan dispatcher so it picks up the new ICD.
        VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);
    }

    // Directly instantiate the Vulkan renderer.
    try {
        gl = new camp::AsyVkRender();
#ifdef HAVE_PTHREAD
        if (gl)
            gl->threadMgr.mainthread = pthread_self();
#endif
        signalRendererReady();
        vulkan = true;
        if (settings::verbose > 1)
            std::cout << "Using Vulkan renderer" << std::endl;
    } catch (const std::exception &e) {
        // Vulkan renderer failed to initialize (e.g., no display, no GPU).
        // Leave gl as nullptr so initRenderer() reports the error.
        std::cerr << "Vulkan renderer initialization failed: " << e.what()
                  << std::endl;
    } catch (...) {
        std::cerr << "Vulkan renderer initialization failed (unknown error)"
                  << std::endl;
    }
#else
#if defined(__APPLE__) && defined(__x86_64__)
    // On Intel Macs, detect Metal availability BEFORE any Vulkan calls.
    // MoltenVK's ICD is found by the Vulkan loader, but if Metal is not
    // available (e.g., headless VM), vkCreateInstance fails with
    // ErrorIncompatibleDriver before we can enumerate devices.
    // Solution: probe Metal directly via dlopen/dlsym, and if unavailable,
    // use a shipped llvmpipe driver found via the Asymptote search path.
    {
        // Step 1: Open Metal.framework
        void *metalHandle = dlopen("/System/Library/Frameworks/Metal.framework/Metal",
                                   RTLD_NOW | RTLD_LOCAL);

        bool metalAvailable = false;
        if (metalHandle) {
            // Step 2: Resolve MTLCreateSystemDefaultDevice
            typedef void *(*PFN_MTLCreateSystemDefaultDevice)();
            PFN_MTLCreateSystemDefaultDevice createDevice =
                reinterpret_cast<PFN_MTLCreateSystemDefaultDevice>(
                    dlsym(metalHandle, "MTLCreateSystemDefaultDevice"));

            if (createDevice) {
                // Step 3: Call it - nil means Metal is not available
                void *device = createDevice();
                if (device != nullptr) {
                    metalAvailable = true;
                }
            }
            dlclose(metalHandle);
        }

        // Step 4: If Metal is NOT available and Vulkan was requested,
        // use the shipped llvmpipe driver from the Asymptote search path.
        if (!metalAvailable && useVulkan) {
            // Locate the shipped llvmpipe ICD JSON via the Asymptote path.
            mem::string icdJsonPath = settings::locateFile("lvp_icd.x86_64.json", true, "");
            std::string icdJsonStr = mem::stdString(icdJsonPath);

            if (!icdJsonStr.empty()) {
                // Convert to absolute path (the Vulkan loader needs it).
                char *absPath = realpath(icdJsonStr.c_str(), nullptr);
                if (absPath != nullptr) {
                    icdJsonStr = absPath;
                    free(absPath);
                }

                setenv("VK_ICD_FILENAMES", icdJsonStr.c_str(), true);
                headlessRenderer = true;

                if (settings::verbose > 1)
                    std::cout << "Metal unavailable on macOS Intel; "
                              << "using llvmpipe fallback (" << icdJsonStr << ")" << std::endl;
            } else {
                if (settings::verbose > 1)
                    std::cout << "Metal unavailable and llvmpipe ICD not found" << std::endl;
            }
        }
    }
#endif // __APPLE__ && __x86_64__

    // If user wants Vulkan, try to load the shared library first.
    if (useVulkan) {
        if (tryLoadVulkanLib()) {
#ifdef __APPLE__
            // GLFW lazily calls dlopen("libvulkan.1.dylib") inside
            // glfwCreateWindowSurface to get vkGetInstanceProcAddr.
            // In a portable/staged build the bundled Vulkan loader's
            // install name is "@executable_path/lib/libvulkan.1.dylib",
            // so that bare-name dlopen fails on a machine without the SDK.
            // Fix: hand GLFW the function pointer from our already-loaded
            // Vulkan loader so it never needs to dlopen it itself.
            {
                PFN_vkGetInstanceProcAddr pfn =
                    (PFN_vkGetInstanceProcAddr)dlsym(
                        vulkanLibHandle, "vkGetInstanceProcAddr");
                if (pfn)
                    glfwInitVulkanLoader(pfn);
            }
#endif // __APPLE__
            return;
        }
        // Vulkan failed to load; fall through to try OpenGL.
        if (settings::verbose > 1)
            std::cout << "Vulkan unavailable, falling back to OpenGL" << std::endl;
    }

    // Try to load the OpenGL renderer.
    if (tryLoadOpenGLLib()) {
        return;
    }
#endif // _WIN32

    // On Unix: both renderers failed to load. Leave gl as nullptr; the error will be
    // reported lazily in initRenderer() when 3D rendering is actually requested.
#ifndef _WIN32
    vulkan = false;
#endif
}

/**
 * Initialise the renderer, optionally selecting based on output format.
 *
 * For WebGL (html) and v3d formats, creates NoRender which requires
 * no GPU libraries - it only sets up state for client-side rendering.
 *
 * For all other formats, lazily calls createRenderer() to probe for Vulkan/
 * OpenGL and instantiate the appropriate renderer.  This defers all GPU
 * library loading until a shipout3() call actually requires rendering.
 *
 * @param format Output format string (e.g., "html", "v3d", or empty/NULL for default)
 */
void initRenderer(const char* format)
{
    // For WebGL and v3d output, use the lightweight NoRender
    // which doesn't require Vulkan or OpenGL libraries
    bool isFormat3D = (format != nullptr &&
                       (strcmp(format, "html") == 0 || strcmp(format, "v3d") == 0));

    // If we have a NoRender but now need GPU rendering (or vice versa),
    // reset and re-initialize with the appropriate renderer
    if (initializedRenderer) {
        bool currentIsNoRender = dynamic_cast<NoRender*>(gl) != nullptr;
        if (currentIsNoRender != isFormat3D) {
            initializedRenderer = false;
            // Clear the old renderer so a new one gets created below.
            // We don't delete it to avoid triggering cleanup code
            // (e.g., glslang::FinalizeProcess()) that can cause issues.
            if (currentIsNoRender)
                gl = nullptr;
        }
    }

    if (initializedRenderer)
        return; // Already fully initialised for this format type

    if (isFormat3D) {
        createNoRenderer();
    } else if (gl == nullptr) {
        // Lazy initialisation: probe for Vulkan/OpenGL only when GPU
        // rendering is actually needed.
#ifdef HAVE_PTHREAD
        // In threaded mode, delegate renderer creation to the glrenderWrapper
        // thread (the OS main thread) to avoid a race condition caused by
        // dlopen from a secondary thread.
        // since gl is still nullptr and we can't use gl->threadMgr yet.
        if(AsyRender::threads) {
            pthread_mutex_lock(&main_wait_mutex);
            pthread_cond_signal(&main_wait_cond);  // Wake glrenderWrapper
            pthread_cond_wait(&main_wait_cond, &main_wait_mutex);  // Wait for done
            pthread_mutex_unlock(&main_wait_mutex);
        } else
#endif
            createRenderer();
    }

    if (gl == nullptr) {
        camp::reportError("No 3D rendering available");
    }

    initializedRenderer = true;

}

} // namespace camp

#else // !HAVE_RENDERER

namespace camp {

[[maybe_unused]] static void signalRendererReady()
{
#ifdef HAVE_PTHREAD
    pthread_mutex_lock(&main_wait_mutex);
    pthread_cond_broadcast(&main_wait_cond);
    pthread_mutex_unlock(&main_wait_mutex);
#endif
}

bool tryLoadVulkan() { return false; }
bool tryLoadOpenGL() { return false; }
void unloadVulkan() {}
void unloadOpenGL() {}
void createRenderer() {}

/**
 * Create a NoRender instance for html/v3d output.
 * This does NOT require Vulkan or OpenGL libraries - it only sets up state
 * needed by jsfile.cc and v3dfile.cc to generate the output files.
 */
#ifdef HAVE_LIBGLM
static void createNoRenderer()
{
    if (gl != nullptr)
        gl = nullptr;

    gl = new camp::NoRender();
    signalRendererReady();
}
#endif

void initRenderer(const char* format)
{
#ifdef HAVE_LIBGLM
    bool isFormat3D = (format != nullptr &&
                       (strcmp(format, "html") == 0 || strcmp(format, "v3d") == 0));

    if (isFormat3D) {
        createNoRenderer();
    }
#endif
    // For non-format3d output without GPU libraries, picture.cc will report
    // a more specific error message.
}

} // namespace camp

#endif // HAVE_RENDERER
