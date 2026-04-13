#pragma once

#ifdef HAVE_VULKAN

#include <GLFW/glfw3.h>
#include <string>

namespace camp
{

/**
 * Virtual interface for renderer callbacks.
 * Derived renderers (AsyVkRender, AsyGlRender) implement this interface
 * to handle GLFW window events in a renderer-specific way.
 */
struct RenderCallbacks
{
    virtual ~RenderCallbacks() = default;

    virtual void onMouseButton(int button, int action, int mods) = 0;
    virtual void onFramebufferResize(int width, int height) = 0;
    virtual void onScroll(double xoffset, double yoffset) = 0;
    virtual void onCursorPos(double xpos, double ypos) = 0;
    virtual void onKey(int key, int scancode, int action, int mods) = 0;
    virtual void onWindowFocus(int focused) = 0;
    virtual void onClose() = 0;
};

// Forward declaration
class AsyRender;

/**
 * Initialize GLFW window with the given dimensions and title.
 * Sets up all callback functions using the provided RenderCallbacks interface.
 */
GLFWwindow* glfwCreateRenderWindow(int width, int height, const std::string& title,
                                    RenderCallbacks* callbacks);

/**
 * Terminate GLFW resources associated with the window.
 */
void glfwDestroyWindow(GLFWwindow* window);

/**
 * Post an empty event to wake up the main loop.
 */
void glfwPostEmptyEventWrapper();

/**
 * Get RenderCallbacks from GLFW window user pointer.
 */
RenderCallbacks* glfwGetCallbacks(GLFWwindow* window);

/**
 * Get action string from button/mods combination.
 * Shared utility for renderer callback implementations.
 */
std::string getGLFWAction(int button, int mods);

} // namespace camp

#endif // HAVE_VULKAN
