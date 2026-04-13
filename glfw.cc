#include "common.h"

#ifdef HAVE_GL

#include "glfw.h"
#include "settings.h"
#include "array.h"

#include <thread>

using settings::getSetting;
using vm::array;
using vm::read;

// Forward declarations for functions defined in glrender.cc
void exitHandler(int);

namespace camp
{

void glfwInitWindow(AsyRender* app, int width, int height, const std::string& title)
{
  // Reset all window hints to defaults before setting OpenGL-specific ones
  glfwDefaultWindowHints();

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
  glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);
  glfwWindowHint(GLFW_TRANSPARENT_FRAMEBUFFER, GLFW_FALSE);

  app->window = glfwCreateWindow(width, height, title.data(), nullptr, nullptr);

  if (app->window == nullptr) {
    cerr << "failed to create a window with width "
         << width << " and height " << height << endl;
    exit(-1);
  }

  glfwMakeContextCurrent(app->window);
  glfwSetWindowUserPointer(app->window, app);
  glfwSetMouseButtonCallback(app->window, AsyRender::mouseButtonCallback);
  glfwSetFramebufferSizeCallback(app->window, AsyRender::framebufferResizeCallback);
  glfwSetScrollCallback(app->window, AsyRender::scrollCallback);
  glfwSetCursorPosCallback(app->window, AsyRender::cursorPosCallback);
  glfwSetKeyCallback(app->window, nullptr);
  glfwSetKeyCallback(app->window, AsyRender::keyCallback);
  glfwSetWindowSizeLimits(app->window, 1, 1, GLFW_DONT_CARE, GLFW_DONT_CARE);
  glfwSetWindowCloseCallback(app->window, AsyRender::closeWindowHandler);
}

void glfwCleanupWindow(AsyRender* app)
{
  if (app->window != nullptr) {
    glfwDestroyWindow(app->window);
    app->window = nullptr;
  }
  glfwTerminate();
}

void glfwPostEmptyEventWrapper()
{
  glfwPostEmptyEvent();
}

AsyRender* glfwGetApp(GLFWwindow* window)
{
  return reinterpret_cast<AsyRender*>(glfwGetWindowUserPointer(window));
}

// Static callback functions defined in AsyRender class but implemented here for clarity
void AsyRender::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
  auto const currentAction = getAction(button, mods);

  if (currentAction.empty())
    return;

  auto app = glfwGetApp(window);

  app->lastAction = currentAction;
}

void AsyRender::framebufferResizeCallback(GLFWwindow* window, int width, int height)
{
  if(width == 0 || height == 0)
    return;

  auto* app = glfwGetApp(window);

  if(width == app->Width && height == app->Height)
    return;

  app->reshape0(width,height);
  app->update();
}

void AsyRender::scrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
  auto app = glfwGetApp(window);
  auto zoomFactor = getSetting<double>("zoomfactor");

  if(zoomFactor > 0.0) {
    if (yoffset > 0)
      app->Zoom *= zoomFactor;
    else
      app->Zoom /= zoomFactor;
  }

  app->update();
}

void AsyRender::cursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{
  static double xprev = 0.0;
  static double yprev = 0.0;

  if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) != GLFW_PRESS)
  {
    xprev = xpos;
    yprev = ypos;
    return;
  }

  auto app = glfwGetApp(window);

  if (app->lastAction == "rotate") {
    // Rotation handled in glrender.cc via currentAction
  }
  else if (app->lastAction == "shift") {
    app->shift(xpos - xprev, ypos - yprev);
    app->update();
  }
  else if (app->lastAction == "pan") {
    if (app->orthographic)
      app->shift(xpos - xprev, ypos - yprev);
    else {
      app->pan(xpos - xprev, ypos - yprev);
    }
    app->update();
  }
  else if (app->lastAction == "zoom") {
    app->zoom(0.0, ypos - yprev);
  }

  xprev = xpos;
  yprev = ypos;
}

void AsyRender::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
  if (action != GLFW_PRESS)
    return;

  auto* app = glfwGetApp(window);

  switch (key)
  {
    case 'H':
      app->home();
      break;
    case 'Q':
      if(!app->Format.empty()) app->exportHandler(0);
      app->quit();
      break;
  }
}

void AsyRender::closeWindowHandler(GLFWwindow *window)
{
  cout << endl;
  exitHandler(0);
}

std::string AsyRender::getAction(int button, int mods)
{
  size_t Button;
  size_t nButtons=5;
  switch(button) {
    case GLFW_MOUSE_BUTTON_LEFT:
      Button=0;
      break;
    case GLFW_MOUSE_BUTTON_MIDDLE:
      Button=1;
      break;
    case GLFW_MOUSE_BUTTON_RIGHT:
      Button=2;
      break;
    default:
      Button=nButtons;
  }

  size_t Mod;
  size_t nMods=4;

  if (mods == 0)
    Mod=0;
  else if(mods & GLFW_MOD_SHIFT)
    Mod=1;
  else if(mods & GLFW_MOD_CONTROL)
    Mod=2;
  else if(mods & GLFW_MOD_ALT)
    Mod=3;
  else
    Mod=nMods;

  if(Button < nButtons) {
    auto left=getSetting<array *>("leftbutton");
    auto middle=getSetting<array *>("middlebutton");
    auto right=getSetting<array *>("rightbutton");
    auto wheelup=getSetting<array *>("wheelup");
    auto wheeldown=getSetting<array *>("wheeldown");
    array *Buttons[]={left,middle,right,wheelup,wheeldown};
    array *a=Buttons[button];
    size_t size=checkArray(a);

    if(Mod < size)
      return read<std::string>(a,Mod);
  }

  return "";
}

void *postEmptyEvent(void *)
{
  glfwPostEmptyEvent();
  return NULL;
}

} // namespace camp

// postEmptyEvent is defined outside the namespace for pthread callback compatibility
void *postEmptyEvent(void *)
{
  glfwPostEmptyEvent();
  return NULL;
}

#endif // HAVE_GL
