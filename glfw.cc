#include "common.h"

#ifdef HAVE_VULKAN

#include "glfw.h"
#include "vkrender.h"
#include "settings.h"

#include <thread>

using settings::getSetting;

// Forward declarations for functions defined in vkrender.cc
void runtimeError(const std::string& s);
void exitHandler(int);

namespace camp
{

void glfwInitWindow(AsyVkRender* app, int width, int height, const std::string& title)
{
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_FOCUSED, GLFW_FALSE);
  glfwWindowHint(GLFW_FOCUS_ON_SHOW, GLFW_FALSE);
  glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

  app->window = glfwCreateWindow(width, height, title.data(), nullptr, nullptr);

  if (app->window == nullptr)
    runtimeError(
      "failed to create a window with width "
      + std::to_string(width) + " and height "
      + std::to_string(height));

  glfwSetWindowUserPointer(app->window, app);
  glfwSetMouseButtonCallback(app->window, AsyVkRender::mouseButtonCallback);
  glfwSetFramebufferSizeCallback(app->window, AsyVkRender::framebufferResizeCallback);
  glfwSetScrollCallback(app->window, AsyVkRender::scrollCallback);
  glfwSetCursorPosCallback(app->window, AsyVkRender::cursorPosCallback);
  glfwSetKeyCallback(app->window, nullptr);
  glfwSetKeyCallback(app->window, AsyVkRender::keyCallback);
  glfwSetWindowSizeLimits(app->window, 1, 1, GLFW_DONT_CARE, GLFW_DONT_CARE);
  glfwSetWindowCloseCallback(app->window, AsyVkRender::closeWindowHandler);
  glfwSetWindowFocusCallback(app->window, AsyVkRender::windowFocusCallback);
}

void glfwCleanupWindow(AsyVkRender* app)
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

AsyVkRender* glfwGetApp(GLFWwindow* window)
{
  return reinterpret_cast<AsyVkRender*>(glfwGetWindowUserPointer(window));
}

// Static callback functions defined in AsyVkRender class but implemented here for clarity
void AsyVkRender::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
  auto const currentAction = getAction(button, mods);

  if (currentAction.empty())
    return;

  auto app = glfwGetApp(window);

  app->lastAction = currentAction;
}

void AsyVkRender::framebufferResizeCallback(GLFWwindow* window, int width, int height)
{
  if(width == 0 || height == 0)
    return;

  auto* app = glfwGetApp(window);

  if(width == app->Width && height == app->Height)
    return;

  app->reshape0(width,height);
  app->update();
  app->remesh=true;
}

void AsyVkRender::scrollCallback(GLFWwindow* window, double xoffset, double yoffset)
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

void AsyVkRender::cursorPosCallback(GLFWwindow* window, double xpos, double ypos)
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

    Arcball arcball(xprev * 2 / app->Width - 1, 1 - yprev * 2 / app->Height, xpos * 2 / app->Width - 1, 1 - ypos * 2 / app->Height);
    triple axis = arcball.axis;
    app->rotateMat = rotate(2 * arcball.angle / app->Zoom * app->ArcballFactor,
                                 glm::dvec3(axis.getx(), axis.gety(), axis.getz())) * app->rotateMat;
    app->update();
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

void AsyVkRender::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
  if (action != GLFW_PRESS)
    return;

  auto* app = glfwGetApp(window);

  switch (key)
  {
    case 'H':
      app->home();
      app->redraw=true;
      break;
    case 'F':
      app->toggleFitScreen();
      break;
    case 'X':
      app->spinx();
      break;
    case 'Y':
      app->spiny();
      break;
    case 'Z':
      app->spinz();
      break;
    case 'S':
      app->idle();
      break;
    case 'M':
      app->cycleMode();
      break;
    case 'E':
      app->queueExport = true;
      break;
    case 'C':
      app->showCamera();
      break;
    case '.': // '>' = '.' + shift
      if (!(mods & GLFW_MOD_SHIFT))
        break;
    case '+':
    case '=':
      app->expand();
      break;
    case ',': // '<' = ',' + shift
      if (!(mods & GLFW_MOD_SHIFT))
        break;
    case '-':
    case '_':
      app->shrink();
      break;
    case 'Q':
      if(!app->Format.empty()) app->exportHandler(0);
      app->quit();
      break;
  }
}

void AsyVkRender::windowFocusCallback(GLFWwindow* window, int focused)
{
    if (focused) {
        // Window gained focus: might need to recreate swapchain
        auto app = glfwGetApp(window);
        app->recreatePipeline = true;
    }
}

void AsyVkRender::closeWindowHandler(GLFWwindow *window)
{
  cout << endl;
  exitHandler(0);
}

std::string AsyVkRender::getAction(int button, int mods)
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
    auto left=getSetting<vm::array *>("leftbutton");
    auto middle=getSetting<vm::array *>("middlebutton");
    auto right=getSetting<vm::array *>("rightbutton");
    auto wheelup=getSetting<vm::array *>("wheelup");
    auto wheeldown=getSetting<vm::array *>("wheeldown");
    vm::array *Buttons[]={left,middle,right,wheelup,wheeldown};
    auto a=Buttons[button];
    size_t size=checkArray(a);

    if(Mod < size)
      return vm::read<std::string>(a,Mod);
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

#endif // HAVE_VULKAN
