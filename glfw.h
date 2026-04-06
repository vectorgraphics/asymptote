#pragma once

#ifdef HAVE_VULKAN

#include <GLFW/glfw3.h>
#include <string>

namespace camp
{

// Forward declaration
class AsyVkRender;

/**
 * Initialize GLFW window with the given dimensions and title.
 * Sets up all callback functions for mouse, keyboard, and window events.
 */
void glfwInitWindow(AsyVkRender* app, int width, int height, const std::string& title);

/**
 * Terminate GLFW resources associated with the window.
 * Called during cleanup/destructor.
 */
void glfwCleanupWindow(AsyVkRender* app);

/**
 * Post an empty event to wake up the main loop.
 * Used for thread synchronization when updates are needed.
 */
void glfwPostEmptyEventWrapper();

/**
 * Get GLFW window user pointer as AsyVkRender*.
 */
AsyVkRender* glfwGetApp(GLFWwindow* window);

} // namespace camp

#endif // HAVE_VULKAN
