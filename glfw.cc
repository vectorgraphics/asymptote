#include "common.h"
#include "vm.h"
#include "array.h"

#ifdef HAVE_VULKAN

#include "glfw.h"
#include "settings.h"

using settings::getSetting;
using vm::read;

// Forward declaration
void runtimeError(const std::string& s);
void exitHandler(int);

namespace camp
{

// Static GLFW callback wrappers that delegate to RenderCallbacks interface
static void onMouseButton(GLFWwindow* window, int button, int action, int mods)
{
    auto callbacks = glfwGetCallbacks(window);
    if (callbacks)
        callbacks->onMouseButton(button, action, mods);
}

static void onFramebufferResize(GLFWwindow* window, int width, int height)
{
    auto callbacks = glfwGetCallbacks(window);
    if (callbacks)
        callbacks->onFramebufferResize(width, height);
}

static void onScroll(GLFWwindow* window, double xoffset, double yoffset)
{
    auto callbacks = glfwGetCallbacks(window);
    if (callbacks)
        callbacks->onScroll(xoffset, yoffset);
}

static void onCursorPos(GLFWwindow* window, double xpos, double ypos)
{
    auto callbacks = glfwGetCallbacks(window);
    if (callbacks)
        callbacks->onCursorPos(xpos, ypos);
}

static void onKey(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    auto callbacks = glfwGetCallbacks(window);
    if (callbacks)
        callbacks->onKey(key, scancode, action, mods);
}

static void onWindowFocus(GLFWwindow* window, int focused)
{
    auto callbacks = glfwGetCallbacks(window);
    if (callbacks)
        callbacks->onWindowFocus(focused);
}

static void onClose(GLFWwindow* window)
{
    auto callbacks = glfwGetCallbacks(window);
    if (callbacks)
        callbacks->onClose();
}

GLFWwindow* glfwCreateRenderWindow(int width, int height, const std::string& title,
                                    RenderCallbacks* callbacks)
{
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_FOCUSED, GLFW_FALSE);
    glfwWindowHint(GLFW_FOCUS_ON_SHOW, GLFW_FALSE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    auto window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);

    if (window == nullptr)
        runtimeError(
            "failed to create a window with width "
            + std::to_string(width) + " and height "
            + std::to_string(height));

    // Store callbacks in window user pointer
    glfwSetWindowUserPointer(window, callbacks);

    // Set up all callbacks
    glfwSetMouseButtonCallback(window, onMouseButton);
    glfwSetFramebufferSizeCallback(window, onFramebufferResize);
    glfwSetScrollCallback(window, onScroll);
    glfwSetCursorPosCallback(window, onCursorPos);
    glfwSetKeyCallback(window, onKey);
    glfwSetWindowSizeLimits(window, 1, 1, GLFW_DONT_CARE, GLFW_DONT_CARE);
    glfwSetWindowCloseCallback(window, onClose);
    glfwSetWindowFocusCallback(window, onWindowFocus);

    return window;
}

void glfwDestroyWindow(GLFWwindow* window)
{
    if (window != nullptr) {
        ::glfwDestroyWindow(window);  // Use global namespace to call GLFW's function
    }
    glfwTerminate();
}

void glfwPostEmptyEventWrapper()
{
    glfwPostEmptyEvent();
}

RenderCallbacks* glfwGetCallbacks(GLFWwindow* window)
{
    return static_cast<RenderCallbacks*>(glfwGetWindowUserPointer(window));
}

// Helper function to get action string from button/mods (uses GLFW constants)
// This is generic code that can be shared between Vulkan and OpenGL renderers
static std::string getActionString(int button, int mods)
{
    size_t Button;
    size_t nButtons = 5;
    switch(button) {
        case GLFW_MOUSE_BUTTON_LEFT:
            Button = 0;
            break;
        case GLFW_MOUSE_BUTTON_MIDDLE:
            Button = 1;
            break;
        case GLFW_MOUSE_BUTTON_RIGHT:
            Button = 2;
            break;
        default:
            Button = nButtons;
    }

    size_t Mod;
    size_t nMods = 4;

    if (mods == 0)
        Mod = 0;
    else if(mods & GLFW_MOD_SHIFT)
        Mod = 1;
    else if(mods & GLFW_MOD_CONTROL)
        Mod = 2;
    else if(mods & GLFW_MOD_ALT)
        Mod = 3;
    else
        Mod = nMods;

    if(Button < nButtons) {
        auto left = getSetting<vm::array *>("leftbutton");
        auto middle = getSetting<vm::array *>("middlebutton");
        auto right = getSetting<vm::array *>("rightbutton");
        auto wheelup = getSetting<vm::array *>("wheelup");
        auto wheeldown = getSetting<vm::array *>("wheeldown");
        vm::array* buttonArray[] = {left, middle, right, wheelup, wheeldown};
        auto a = buttonArray[Button];
        size_t size = checkArray(a);

        if(Mod < size)
            return read<std::string>(a, Mod);
    }

    return "";
}

// Export getActionString for use in renderer implementations
std::string getGLFWAction(int button, int mods)
{
    return getActionString(button, mods);
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
