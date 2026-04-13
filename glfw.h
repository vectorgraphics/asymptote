#pragma once

#ifdef HAVE_GL

#include <GLFW/glfw3.h>
#include <string>

namespace camp
{

class AsyRender
{
public:
  GLFWwindow* window = nullptr;

  int Width, Height;
  double Zoom;
  double ArcballFactor;
  bool orthographic;
  std::string Format;

  std::string lastAction = "";

  // Static callback functions (implemented in glfw.cc)
  static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
  static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
  static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
  static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos);
  static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
  static void closeWindowHandler(GLFWwindow *window);

  // Virtual methods to be implemented by renderer-specific classes
  virtual void reshape0(int width, int height) = 0;
  virtual void update() = 0;
  virtual void shift(double dx, double dy) = 0;
  virtual void pan(double dx, double dy) = 0;
  virtual void zoom(double dx, double dy) = 0;
  virtual void home() = 0;
  virtual void quit() = 0;
  virtual void exportHandler(int=0) = 0;

  // Helper function for getting action from button/mods
  static std::string getAction(int button, int mods);

  virtual ~AsyRender() = default;
};

/**
 * Initialize GLFW window with the given dimensions and title.
 * Sets up all callback functions for mouse, keyboard, and window events.
 */
void glfwInitWindow(AsyRender* app, int width, int height, const std::string& title);

/**
 * Terminate GLFW resources associated with the window.
 * Called during cleanup/destructor.
 */
void glfwCleanupWindow(AsyRender* app);

/**
 * Post an empty event to wake up the main loop.
 * Used for thread synchronization when updates are needed.
 */
void glfwPostEmptyEventWrapper();

/**
 * Get GLFW window user pointer as AsyRender*.
 */
AsyRender* glfwGetApp(GLFWwindow* window);

} // namespace camp

#endif // HAVE_GL
