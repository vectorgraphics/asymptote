#include <limits>
#include <chrono>
#include <thread>

#include "vkrender.h"
#include "shaderResources.h"
#include "picture.h"
#include "drawimage.h"
#include "EXRFiles.h"
#include "fpu.h"

#include "vkutils.h"
#include "ThreadSafeQueue.h"

// For debugging:
#if defined(ENABLE_VK_VALIDATION)
#define VALIDATION
#endif

#define SHADER_DIRECTORY "shaders/"
#define VALIDATION_LAYER "VK_LAYER_KHRONOS_validation"

#define VARIABLE_NAME(var) (#var)

#if defined(_WIN32)
#include <Windows.h>
#include <vulkan/vulkan_win32.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#endif

using settings::getSetting;
using settings::Setting;
using namespace glm;

static size_t timeout=1000000000;

void exitHandler(int);

void runtimeError(std::string s)
{
  cerr << "error: " << s << endl;
  exit(-1);
}

#ifdef HAVE_VULKAN
uint32_t apiVersion=VK_API_VERSION_1_4;

std::vector<const char*> instanceExtensions
{
  VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
  // VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME,
#if defined(VALIDATION) || defined(DEBUG)
  VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
  VK_EXT_DEBUG_REPORT_EXTENSION_NAME
#endif

};
#endif

#ifdef HAVE_LIBGLM

namespace camp
{
dmat4 projViewMat;
dmat4 normMat;

const Int timePartialSumVerbosity=4;

std::vector<char> readFile(const std::string& filename)
{
  std::ifstream file(filename, std::ios::ate | std::ios::binary);
  if (!file.is_open())
    runtimeError("failed to open file " + filename);

  size_t fileSize = (size_t) file.tellg();
  std::vector<char> buffer(fileSize);

  file.seekg(0);
  file.read(buffer.data(), fileSize);
  file.close();

  return buffer;
}

#ifdef HAVE_VULKAN
void closeWindowHandler(GLFWwindow *);

SwapChainDetails::SwapChainDetails(
  vk::PhysicalDevice gpu,
  vk::SurfaceKHR surface) :
  capabilities {gpu.getSurfaceCapabilitiesKHR(surface)},
  formats {gpu.getSurfaceFormatsKHR(surface)},
  presentModes {gpu.getSurfacePresentModesKHR(surface)}
{ }

SwapChainDetails::operator bool() const
{
  return !formats.empty() && !presentModes.empty();
}

vk::SurfaceFormatKHR
SwapChainDetails::chooseSurfaceFormat() const
{
  for (const auto& availableFormat : formats) {
    if (availableFormat.format == vk::Format::eB8G8R8A8Unorm &&
        availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
      return availableFormat;
    }
  }

  return formats.front();
}

vk::PresentModeKHR
SwapChainDetails::choosePresentMode() const
{
  bool vsync=settings::getSetting<bool>("vsync");
  for (const auto& mode : presentModes) {
    if ((!vsync && mode == vk::PresentModeKHR::eImmediate) ||
        (vsync && mode == vk::PresentModeKHR::eFifo)) {
      return mode;
    }
  }

  return presentModes.front();
}

vk::Extent2D
SwapChainDetails::chooseExtent(size_t width, size_t height) const
{
  if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
    return capabilities.currentExtent;
  }

  auto extent = vk::Extent2D(
    static_cast<uint32_t>(width),
    static_cast<uint32_t>(height)
  );

  extent.width = clamp(
                  extent.width,
                  capabilities.minImageExtent.width,
                  capabilities.maxImageExtent.width
                 );
  extent.height = clamp(
                    extent.height,
                    capabilities.minImageExtent.height,
                    capabilities.maxImageExtent.height
                  );

  return extent;
}

std::uint32_t
SwapChainDetails::chooseImageCount() const
{
  auto imageCount = capabilities.minImageCount + 1;

  if(capabilities.maxImageCount > 0 &&
     imageCount > capabilities.maxImageCount) {
    imageCount = capabilities.maxImageCount;
  }

  return imageCount;
}

void AsyVkRender::setDimensions(int width, int height, double x, double y)
{
  double aspect = ((double) width) / height;
  double xshift = (x / (double) width + Shift.getx() * Xfactor) * Zoom;
  double yshift = (y / (double) height + Shift.gety() * Yfactor) * Zoom;
  double zoominv = 1.0 / Zoom;
  if (orthographic) {
    double xsize = Xmax - Xmin;
    double ysize = Ymax - Ymin;
    if (xsize < ysize * aspect) {
      double r = 0.5 * ysize * aspect * zoominv;
      double X0 = 2.0 * r * xshift;
      double Y0 = ysize * zoominv * yshift;
      xmin = -r - X0;
      xmax = r - X0;
      ymin = Ymin * zoominv - Y0;
      ymax = Ymax * zoominv - Y0;
    } else {
      double r = 0.5 * xsize * zoominv / aspect;
      double X0 = xsize * zoominv * xshift;
      double Y0 = 2.0 * r * yshift;
      xmin = Xmin * zoominv - X0;
      xmax = Xmax * zoominv - X0;
      ymin = -r - Y0;
      ymax = r - Y0;
    }
  } else {
    double r = H * zoominv;
    double rAspect = r * aspect;
    double X0 = 2.0 * rAspect * xshift;
    double Y0 = 2.0 * r * yshift;
    xmin = -rAspect - X0;
    xmax = rAspect - X0;
    ymin = -r - Y0;
    ymax = r - Y0;
  }
}

void AsyVkRender::setProjection()
{
  setDimensions(width, height, X, Y);

  if(orthographic) vk->ortho(xmin,xmax,ymin,ymax,-Zmax,-Zmin);
  else vk->frustum(xmin,xmax,ymin,ymax,-Zmax,-Zmin);
  newUniformBuffer = true;
}

void AsyVkRender::updateModelViewData()
{
  normMat = inverse(viewMat);
  newUniformBuffer = true;
}

void *postEmptyEvent(void *)
{
  glfwPostEmptyEvent();
  return NULL;
}

void AsyVkRender::update()
{
  capzoom();

  double cz = 0.5 * (Zmin + Zmax);
  viewMat = translate(translate(dmat4(1.0), dvec3(cx, cy, cz)) * rotateMat, dvec3(0, 0, -cz));

  setProjection();
  updateModelViewData();

#ifdef HAVE_PTHREAD
  if(View) {
    pthread_t postThread;
    if(pthread_create(&postThread,NULL,postEmptyEvent,NULL) == 0)
      pthread_join(postThread,NULL);
  }
#endif
  redraw=true;
}

#endif

double AsyVkRender::getRenderResolution(triple Min) const
{
  double prerender = settings::getSetting<double>("prerender");

  if (prerender <= 0.0)
    return 0.0;

  prerender = 1.0 / prerender;
  double perspective = orthographic ? 0.0 : 1.0 / Zmax;
  double s = perspective ? Min.getz() * perspective : 1.0;
  triple b(Xmin, Ymin, Zmin);
  triple B(Xmax, Ymax, Zmax);
  pair size3(s * (B.getx() - b.getx()), s * (B.gety() - b.gety()));
  pair size2(width, height);
  return prerender * size3.length() / size2.length();
}

#ifdef HAVE_VULKAN

void AsyVkRender::initWindow()
{
  if (!window) {
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_FOCUSED, GLFW_FALSE);
    glfwWindowHint(GLFW_FOCUS_ON_SHOW, GLFW_FALSE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    window = glfwCreateWindow(width, height, title.data(), nullptr, nullptr);

    if (window == nullptr)
      runtimeError(
        "failed to create a window with width "
        + std::to_string(width) + " and height "
        + std::to_string(height));
  }

  glfwSetWindowUserPointer(window, this);
  glfwSetMouseButtonCallback(window, mouseButtonCallback);
  glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
  glfwSetScrollCallback(window, scrollCallback);
  glfwSetCursorPosCallback(window, cursorPosCallback);
  glfwSetKeyCallback(window, NULL);
  glfwSetKeyCallback(window, keyCallback);
  glfwSetWindowSizeLimits(window, 1, 1, GLFW_DONT_CARE, GLFW_DONT_CARE);
  glfwSetWindowCloseCallback(window,closeWindowHandler);
  glfwSetWindowFocusCallback(window,windowFocusCallback);
}

void AsyVkRender::updateHandler(int) {
  if(vk->View && !interact::interactive) {
    glfwHideWindow(vk->window);
    if(!getSetting<bool>("fitscreen"))
      vk->Fitscreen=0;
  }

  if(vk->device)
    vk->device->waitIdle();
  vk->resize=true;
  vk->redisplay=true;
  vk->redraw=true;
  vk->remesh=true;
  vk->waitEvent=false;
  vk->recreatePipeline=true;
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
    auto left=settings::getSetting<vm::array *>("leftbutton");
    auto middle=settings::getSetting<vm::array *>("middlebutton");
    auto right=settings::getSetting<vm::array *>("rightbutton");
    auto wheelup=settings::getSetting<vm::array *>("wheelup");
    auto wheeldown=settings::getSetting<vm::array *>("wheeldown");
    vm::array *Buttons[]={left,middle,right,wheelup,wheeldown};
    auto a=Buttons[button];
    size_t size=checkArray(a);

    if(Mod < size)
      return vm::read<std::string>(a,Mod);
  }

  return "";
}

void AsyVkRender::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
  auto const currentAction = getAction(button, mods);

  if (currentAction.empty())
    return;

  auto app = reinterpret_cast<AsyVkRender*>(glfwGetWindowUserPointer(window));

  app->lastAction = currentAction;
}

void AsyVkRender::framebufferResizeCallback(GLFWwindow* window, int width, int height)
{
  if(width == 0 || height == 0)
    return;

  auto* app = static_cast<AsyVkRender*>(glfwGetWindowUserPointer(window));

  if(width == app->width && height == app->height)
    return;

  app->reshape0(width,height);
  app->update();
  app->remesh=true;
}

void AsyVkRender::scrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
  auto app = reinterpret_cast<AsyVkRender*>(glfwGetWindowUserPointer(window));
  auto zoomFactor = settings::getSetting<double>("zoomfactor");

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

  auto app = reinterpret_cast<AsyVkRender*>(glfwGetWindowUserPointer(window));

  if (app->lastAction == "rotate") {

    Arcball arcball(xprev * 2 / app->width - 1, 1 - yprev * 2 / app->height, xpos * 2 / app->width - 1, 1 - ypos * 2 / app->height);
    triple axis = arcball.axis;
    app->rotateMat = rotate(2 * arcball.angle / app->Zoom * app->ArcballFactor,
                                 dvec3(axis.getx(), axis.gety(), axis.getz())) * app->rotateMat;
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

  auto app = reinterpret_cast<AsyVkRender*>(glfwGetWindowUserPointer(window));

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
    case 'P':
      if(settings::getSetting<bool>("reverse")) app->Animate=false;
      settings::Setting("reverse")=app->Step=false;
      app->animate();
      break;
    case 'R':
      if(!settings::getSetting<bool>("reverse")) app->Animate=false;
      settings::Setting("reverse")=true;
      app->Step=false;
      app->animate();
      break;
    case ' ':
      app->Step=true;
      app->animate();
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
        auto app = reinterpret_cast<AsyVkRender*>(glfwGetWindowUserPointer(window));
        app->recreatePipeline = true;
    }
}

AsyVkRender::~AsyVkRender()
{
  if (this->View) {
    glfwDestroyWindow(this->window);
    glfwTerminate();
  }

  glslang::FinalizeProcess();
}

#endif

bool ispow2(unsigned int n) {return n > 0 && !(n & (n - 1));}
void checkpow2(unsigned int n, std::string s) {
  if(!ispow2(n)) {
    runtimeError(s+" must be a power of two");
    exit(-1);
  }
}

void AsyVkRender::vkrender(VkrenderFunctionArgs const& args)
{
#if !defined(_WIN32)
      setenv("XMODIFIERS","",true);
#endif

  bool v3d=args.format == "v3d";
  bool webgl=args.format == "html";
  bool format3d=webgl || v3d;

  this->pic = args.pic;
  this->Prefix=args.prefix;
  this->Format = args.format;
  this->remesh = true;
  this->nlights = args.nlightsin;
  this->Lights = args.lights;
  this->LightsDiffuse = args.diffuse;
  this->Oldpid = args.oldpid;

  this->Angle = args.angle * radians;
  this->lastzoom = 0;
  this->Zoom0 = args.zoom;
  this->Shift = args.shift / args.zoom;
  this->Margin = args.margin;

  for (int i = 0; i < 4; i++)
    this->Background[i] = static_cast<float>(args.background[i]);

  this->ViewExport=args.view;

  this->View = args.view && !settings::getSetting<bool>("offscreen");

  this->title = std::string(PACKAGE_NAME)+": "+ args.prefix.c_str();

  Xmin = args.m.getx();
  Xmax = args.M.getx();
  Ymin = args.m.gety();
  Ymax = args.M.gety();
  Zmin = args.m.getz();
  Zmax = args.M.getz();

  orthographic = (this->Angle == 0.0);
  H = orthographic ? 0.0 : -tan(0.5 * this->Angle) * Zmax;
  Xfactor = Yfactor = 1.0;

#ifdef HAVE_PTHREAD
  static bool initializedView=false;
  if(vkinitialize)
    Fitscreen=1;
#endif

  for(int i=0; i < 16; ++i)
    T[i]=args.t[i];

  for(int i=0; i < 16; ++i)
    Tup[i]=args.tup[i];

  if(!(initialized && (interact::interactive ||
                       settings::getSetting<bool>("animating")))) {
    antialias=settings::getSetting<Int>("antialias") > 1;
    double expand;
    if(format3d)
      expand=1.0;
    else {
      expand=settings::getSetting<double>("render");
      if(expand < 0)
        expand *= (Format.empty() || Format == "eps" || Format == "pdf")                 ? -2.0 : -1.0;
      if(antialias) expand *= 2.0;
    }

    oWidth=args.width;
    oHeight=args.height;
    Aspect=args.width/args.height;

    fullWidth=(int) ceil(expand*args.width);
    fullHeight=(int) ceil(expand*args.height);

    if(format3d) {
      width=fullWidth;
      height=fullHeight;
    } else {
#ifdef HAVE_VULKAN
      GLFWmonitor* monitor=NULL;
      glfwInit();
      monitor=glfwGetPrimaryMonitor();
      if(monitor) {
        int mx, my;
        glfwGetMonitorWorkarea(monitor, &mx, &my, &screenWidth, &screenHeight);
      } else
#endif
        {
          screenWidth=width;
          screenHeight=height;
        }

      width=min(fullWidth,screenWidth);
      height=min(fullHeight,screenHeight);

      if(width > height*Aspect)
        width=min((int) (ceil(height*Aspect)),screenWidth);
      else
        height=min((int) (ceil(width/Aspect)),screenHeight);
    }

#ifdef HAVE_VULKAN
    home(format3d);
#endif
    if(format3d) {
      remesh=true;
      return;
    }
    maxFragments=0;

    ArcballFactor=1+8.0*hypot(Margin.getx(),Margin.gety())/hypot(width,height);
    Aspect=((double) width)/height;

#ifdef HAVE_VULKAN
    setosize();
#endif
  }

#ifdef HAVE_VULKAN
  havewindow=initialized && vkthread;

  if(vkthread && format3d)
    format3dWait=true;

  clearMaterials();
  this->shouldUpdateBuffers = true;
  initialized=true;
#endif

#ifdef HAVE_PTHREAD
  if(vkthread && initializedView) {
    if(View) {
      // called from asymain thread, main thread handles vulkan rendering
      hideWindow=false;
      messageQueue.enqueue(updateRenderer);
      clearBuffers();
    } else readyAfterExport=queueExport=true;
    return;
  }
#endif

  GPUcompress=settings::getSetting<bool>("GPUcompress");

  localSize=settings::getSetting<Int>("GPUlocalSize");
  checkpow2(localSize,"GPUlocalSize");
  blockSize=settings::getSetting<Int>("GPUblockSize");
  checkpow2(blockSize,"GPUblockSize");
  groupSize=localSize*blockSize;

#ifdef HAVE_VULKAN
  if(vkinitialize) {
    interlock=settings::getSetting<bool>("GPUinterlock");
    fxaa=settings::getSetting<bool>("fxaa");
    srgb=settings::getSetting<bool>("srgb");

    Animate=settings::getSetting<bool>("autoplay") && vkthread;
    ibl=settings::getSetting<bool>("ibl");
  }

  if(View) {
    if(vkinitialize)
      initWindow();
    if(!getSetting<bool>("fitscreen"))
      Fitscreen=0;
    fitscreen();
    Aspect=((double) width)/height;
    setosize();
    initializedView=true;
  }

  if(vkinitialize) {
    vkinitialize=false;
    initVulkan();
  }

  readyForUpdate=true;
  mainLoop();
#endif
}

#ifdef HAVE_VULKAN
void AsyVkRender::initVulkan()
{
#ifdef __APPLE__
  setenv("MVK_CONFIG_LOG_LEVEL","0",false);

  // Prefer low power mode for better stability: TODO Remove
  setenv("MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS", "1", true);
  setenv("MVK_CONFIG_PERFORMANCE_TRACKING", "0", true);
#endif

  VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

  if (!glslang::InitializeProcess())
    runtimeError("failed to initialize glslang");

  maxFramesInFlight=View ? settings::getSetting<Int>("maxFramesInFlight") : 1;
  frameObjects.resize(maxFramesInFlight);

  if (settings::verbose > 1) {
    std::cout << "Using " << maxFramesInFlight
              << " maximum frame(s) in flight" << std::endl;
  }
  createInstance();
  createDebugMessenger();
  if (View) createSurface();
  pickPhysicalDevice();

  fpu_trap(false); // Work around FE_INVALID.

  createLogicalDevice();
  createAllocator();
  createCommandPools();
  createCommandBuffers();
  if (View) createSwapChain();
  else createOffscreenBuffers();

  if (fxaa)
  {
    setupPostProcessingComputeParameters();
  }
  createImageViews();
  createSyncObjects();

  createDescriptorSetLayout();
  createComputeDescriptorSetLayout();

  createBuffers();

  if (ibl) {
    initIBL();
  }

  createDescriptorPool();
  createComputeDescriptorPool();
  createDescriptorSets();

  createImmediateRenderTargets();
  writeDescriptorSets();
  writeMaterialAndLightDescriptors();

  createCountRenderPass();
  createGraphicsRenderPass();
  createGraphicsPipelineLayout();
  createGraphicsPipelines();

  createComputePipelines();// gpu indexing + post processing
  fpu_trap(settings::trap()); // Work around FE_INVALID.

  createAttachments();
  createFramebuffers();
  createExportResources();
}

void AsyVkRender::recreateSwapChain()
{
  device->waitIdle();

  // Reset timeline semaphore values to avoid timeout issues
  currentTimelineValue = 0;
  for (auto& frameObj : frameObjects) {
    frameObj.timelineValue = 0;
  }

  resetDepth=true;

  createSwapChain();

  if (fxaa)
  {
    setupPostProcessingComputeParameters();
  }
  createImmediateRenderTargets();
  writeDescriptorSets();
  writeMaterialAndLightDescriptors();
  createImageViews();
  createSyncObjects();
  createCountRenderPass();
  createGraphicsRenderPass();
  createGraphicsPipelines();
  createAttachments();
  createFramebuffers();
  createExportResources();

  redisplay=true;
  waitEvent=false;
}

void AsyVkRender::zeroTransparencyBuffers()
{
  auto const clearCmdBuffer=beginSingleCommands();
  zeroBuffer(clearCmdBuffer,globalSumBf.getBuffer());
  zeroBuffer(clearCmdBuffer,opaqueDepthBf.getBuffer());
  if(GPUcompress)
    zeroBuffer(clearCmdBuffer,indexBf.getBuffer());
  else
    zeroBuffer(clearCmdBuffer,countBf.getBuffer());
  endSingleCommands(clearCmdBuffer);
}

std::set<std::string> AsyVkRender::getInstanceExtensions()
{
  std::set<std::string> extensions;
  auto availableExtensions = vk::enumerateInstanceExtensionProperties();
  for (auto& extension : availableExtensions) {
    extensions.insert(extension.extensionName);
  }
  return extensions;
}

std::set<std::string> AsyVkRender::getDeviceExtensions(vk::PhysicalDevice& device)
{
  std::set<std::string> extensions;
  auto availableExtensions = device.enumerateDeviceExtensionProperties();
  for (auto& extension : availableExtensions) {
    extensions.insert(extension.extensionName);
  }
  return extensions;
}

std::vector<const char*> AsyVkRender::getRequiredInstanceExtensions()
{
  uint32_t glfwExtensionCount;
  auto const glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
  std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

  for(auto& extension : instanceExtensions) {
    extensions.emplace_back(extension);
  }

  return extensions;
}

void AsyVkRender::createInstance()
{
  auto appInfo = vk::ApplicationInfo(
    PACKAGE_STRING,
    VK_MAKE_VERSION(1, 0, 0),
    "No Engine",
    VK_MAKE_VERSION(1, 0, 0),
    apiVersion
  );
  auto supportedExtensions = getInstanceExtensions();
  auto supportedLayers = vk::enumerateInstanceLayerProperties();
  auto extensions = getRequiredInstanceExtensions();

  auto isLayerSupported = [supportedLayers](std::string layerName) {
    return std::find_if(
      supportedLayers.begin(),
      supportedLayers.end(),
      [layerName](vk::LayerProperties const& layer) {
        return layer.layerName.data() == layerName;
      }) != supportedLayers.end();
  };

  auto isExtensionSupported = [supportedExtensions](std::string extension) {
    return std::find_if(
      supportedExtensions.begin(),
      supportedExtensions.end(),
      [extension](std::string const& supportedExt) {
        return supportedExt == extension;
      }) != supportedExtensions.end();
  };

#ifdef VALIDATION
  if (isLayerSupported(VALIDATION_LAYER)) {
    validationLayers.emplace_back(VALIDATION_LAYER);
  } else if (settings::verbose > 1) {
    std::cout << "Validation layers are not supported by the current Vulkan instance" << std::endl;
  }
#endif

  std::vector<const char*> all_extensions;
  all_extensions.reserve(supportedExtensions.size());

  for (const auto& str : supportedExtensions) {
      all_extensions.push_back(str.c_str());
  }

  auto const instanceCI = vk::InstanceCreateInfo(
#if defined(__APPLE__)
    vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR,
#else
    {},
#endif
    &appInfo,
    VEC_VIEW(validationLayers),
    VEC_VIEW(all_extensions)
  );
  instance = vk::createInstanceUnique(instanceCI);
  VULKAN_HPP_DEFAULT_DISPATCHER.init(*instance);
}

void AsyVkRender::createDebugMessenger()
{
#if defined(VALIDATION)
  vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
  vk::DebugUtilsMessageTypeFlagsEXT typeFlags(vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);
  if (settings::verbose > 2)
  {
    severityFlags |= vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning;
    typeFlags |= vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral;
  }
  if (settings::verbose > 2)
  {
    severityFlags |= vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo;
    typeFlags |= typeFlags |= vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;
  }
  if (settings::verbose > 2)
  {
    severityFlags |= vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose;
  }

  auto const debugCreateInfo = vk::DebugUtilsMessengerCreateInfoEXT(
          {},
          severityFlags,
          typeFlags,
          [](VkDebugUtilsMessageSeverityFlagBitsEXT msgSeverity,
             VkDebugUtilsMessageTypeFlagsEXT msgType,
             VkDebugUtilsMessengerCallbackDataEXT const* pCallbackData,
             void* pUserData) -> VkBool32 {
            switch (static_cast<vk::DebugUtilsMessageSeverityFlagBitsEXT>(msgSeverity))
            {
              case vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo:
                cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
                break;
              case vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose:
                cerr << "[VERBOSE] validation layer: " << pCallbackData->pMessage << std::endl;
                break;
              case vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning:
              case vk::DebugUtilsMessageSeverityFlagBitsEXT::eError:
                reportWarning(pCallbackData->pMessage);
                break;
              default:
                break;
            }

            return vk::False;
          },
          this
  );
  instance->createDebugUtilsMessengerEXTUnique(debugCreateInfo);
#endif
}

void AsyVkRender::createSurface()
{
#if defined(_WIN32)
  vk::Win32SurfaceCreateInfoKHR createInfo = {};
  createInfo.hwnd = glfwGetWin32Window(window);
  createInfo.hinstance = GetModuleHandleA(nullptr);

  vk::SurfaceKHR tmpSurface;

  vkutils::checkVkResult(instance->createWin32SurfaceKHR(
    &createInfo,
    nullptr,
    &tmpSurface
  ));

  surface=vk::UniqueSurfaceKHR(tmpSurface);
#else
  VkSurfaceKHR surfaceTmp;
  if (glfwCreateWindowSurface(*instance, window, nullptr, &surfaceTmp) != VK_SUCCESS)
    runtimeError("failed to create window surface");
  surface=vk::UniqueSurfaceKHR(surfaceTmp, *instance);
#endif
}

void AsyVkRender::createAllocator()
{
  VmaVulkanFunctions vkFuncs = {};
  vkFuncs.vkGetInstanceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr;
  vkFuncs.vkGetDeviceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr;
  vkFuncs.vkGetBufferMemoryRequirements2KHR = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetBufferMemoryRequirements2 ? VULKAN_HPP_DEFAULT_DISPATCHER.vkGetBufferMemoryRequirements2 : VULKAN_HPP_DEFAULT_DISPATCHER.vkGetBufferMemoryRequirements2KHR;
  vkFuncs.vkGetImageMemoryRequirements2KHR = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetImageMemoryRequirements2 ? VULKAN_HPP_DEFAULT_DISPATCHER.vkGetImageMemoryRequirements2 : VULKAN_HPP_DEFAULT_DISPATCHER.vkGetImageMemoryRequirements2KHR;
  vkFuncs.vkBindBufferMemory2KHR = VULKAN_HPP_DEFAULT_DISPATCHER.vkBindBufferMemory2 ? VULKAN_HPP_DEFAULT_DISPATCHER.vkBindBufferMemory2 : VULKAN_HPP_DEFAULT_DISPATCHER.vkBindBufferMemory2KHR;
  vkFuncs.vkBindImageMemory2KHR = VULKAN_HPP_DEFAULT_DISPATCHER.vkBindImageMemory2 ? VULKAN_HPP_DEFAULT_DISPATCHER.vkBindImageMemory2 : VULKAN_HPP_DEFAULT_DISPATCHER.vkBindImageMemory2KHR;

  VmaAllocatorCreateInfo createInfo = {};
  createInfo.vulkanApiVersion=apiVersion;
  createInfo.physicalDevice = physicalDevice;
  createInfo.device = *device;
  createInfo.instance = *instance;
  createInfo.pVulkanFunctions = &vkFuncs;

  allocator = vma::cxx::UniqueAllocator(createInfo);
}

void AsyVkRender::pickPhysicalDevice()
{
  bool remote=false;

  if(View) {
    char *display=getenv("DISPLAY");
    remote=display ? string(display).find(":") != 0 : false;
  }

  Int device=getSetting<Int>("device");

  ssize_t count=0;
  bool showDevices=settings::verbose > 1;
  if(device >= 0 || showDevices) {
    for(auto& dev: instance->enumeratePhysicalDevices()) {
      if(showDevices)
        std::cerr << "Device " << count << ": " << dev.getProperties().deviceName << std::endl;
      count++;
    }
  }

  bool software=View && remote;

  if(device >= 0 && device < count) {
    physicalDevice=instance->enumeratePhysicalDevices()[device];
    if(software && physicalDevice.getProperties().deviceType !=
       vk::PhysicalDeviceType::eCpu)
      runtimeError("remote onscreen rendering requires the llvmpipe device");
  } else {
    auto const getDeviceScore =
      [this,software](vk::PhysicalDevice& device) -> size_t
      {
        size_t score = 0u;

        if (!this->isDeviceSuitable(device))
          return score;

        auto const msaa = getMaxMSAASamples(device).second;

        switch (msaa)
          {
            case vk::SampleCountFlagBits::e64:
            case vk::SampleCountFlagBits::e32:
            case vk::SampleCountFlagBits::e16:

              score += 10;
              break;

            case vk::SampleCountFlagBits::e8:
            case vk::SampleCountFlagBits::e4:
            case vk::SampleCountFlagBits::e2:

              score += 5;
              break;

            default:

              break;
          }

        auto const props = device.getProperties();

        if(vk::PhysicalDeviceType::eDiscreteGpu == props.deviceType) {
          if(software) return 0;
          score += 10;
        } else if(vk::PhysicalDeviceType::eIntegratedGpu == props.deviceType) {
          if(software) return 0;
          score += 5;
        } else if(vk::PhysicalDeviceType::eCpu == props.deviceType &&
                  software) {
          // Force using software renderer for remote onscreen rendering
          score += 100;
        }

        return score;
      };

    std::pair<size_t, vk::PhysicalDevice> highestDeviceScore;

    for (auto & dev: instance->enumeratePhysicalDevices())
      {
        auto const score = getDeviceScore(dev);

        if (nullptr == highestDeviceScore.second
            || score > highestDeviceScore.first)
          highestDeviceScore = std::make_pair(score, dev);
      }

    if (0 == highestDeviceScore.first)
      runtimeError("no suitable GPUs");

    physicalDevice = highestDeviceScore.second;
  }

  if(settings::verbose > 1)
    cout << "Using device " << physicalDevice.getProperties().deviceName
         << endl;

  std::uint32_t nSamples;

  std::tie(nSamples, msaaSamples) = getMaxMSAASamples(physicalDevice);

  if(settings::verbose > 1 && msaaSamples != vk::SampleCountFlagBits::e1)
    cout << "Multisampling enabled with sample width " << nSamples
         << endl;
}

std::pair<std::uint32_t, vk::SampleCountFlagBits>
AsyVkRender::getMaxMSAASamples( vk::PhysicalDevice & gpu )
{
  // FXAA means we disable MSAA
  if (settings::getSetting<bool>("fxaa"))
  {
    return std::make_pair(1, vk::SampleCountFlagBits::e1);
  }

  vk::PhysicalDeviceProperties props { };

  gpu.getProperties( &props );

  auto const count = props.limits.framebufferColorSampleCounts & props.limits.framebufferDepthSampleCounts;
  auto const maxSamples = settings::getSetting<Int>("multisample");

  if (count & vk::SampleCountFlagBits::e64 && maxSamples >= 64)
    return std::make_pair(64, vk::SampleCountFlagBits::e64);
  if (count & vk::SampleCountFlagBits::e32 && maxSamples >= 32)
    return std::make_pair(32, vk::SampleCountFlagBits::e32);
  if (count & vk::SampleCountFlagBits::e16 && maxSamples >= 16)
    return std::make_pair(16, vk::SampleCountFlagBits::e16);
  if (count & vk::SampleCountFlagBits::e8 && maxSamples >= 8)
    return std::make_pair(8, vk::SampleCountFlagBits::e8);
  if (count & vk::SampleCountFlagBits::e4 && maxSamples >= 4)
    return std::make_pair(4, vk::SampleCountFlagBits::e4);
  if (count & vk::SampleCountFlagBits::e2 && maxSamples >= 2)
    return std::make_pair(2, vk::SampleCountFlagBits::e2);

  return std::make_pair(1, vk::SampleCountFlagBits::e1);
}

QueueFamilyIndices AsyVkRender::findQueueFamilies(vk::PhysicalDevice& physicalDevice, vk::SurfaceKHR* surface)
{
  QueueFamilyIndices indices;

  auto queueFamilies = physicalDevice.getQueueFamilyProperties();

  for (auto u = 0u; u < queueFamilies.size(); u++)
  {
    auto const & family = queueFamilies[u];

    if (family.queueFlags & vk::QueueFlagBits::eGraphics) {
      indices.renderQueueFamily = u,
      indices.renderQueueFamilyFound = true;
    }

    if (surface != nullptr && VK_FALSE != physicalDevice.getSurfaceSupportKHR(u, *surface)) {
      indices.presentQueueFamily = u,
      indices.presentQueueFamilyFound = true;
    }

    if (family.queueFlags & vk::QueueFlagBits::eTransfer) {
      indices.transferQueueFamily = u,
      indices.transferQueueFamilyFound = true;
    }
  }

  return indices;
}

bool AsyVkRender::isDeviceSuitable(vk::PhysicalDevice& device)
{
  auto const indices = findQueueFamilies(device, View ? &*surface : nullptr);
  if (!indices.transferQueueFamilyFound
      || !indices.renderQueueFamilyFound
      || !(indices.presentQueueFamilyFound || !View))
      return false;

  if (!checkDeviceExtensionSupport(device))
    return false;

  auto const features = device.getFeatures();

  if (!View) {
    return features.samplerAnisotropy;
  }

  auto const swapDetails = SwapChainDetails(device, *surface);

  if (View && !swapDetails) {
    return false;
  }

  return features.samplerAnisotropy;
}

bool AsyVkRender::checkDeviceExtensionSupport(vk::PhysicalDevice& device)
{
  auto extensions = device.enumerateDeviceExtensionProperties();
  std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());
  if (View) requiredExtensions.insert(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

  for (auto& extension : extensions) {
    requiredExtensions.erase(extension.extensionName);
  }
  return requiredExtensions.empty();
}

void AsyVkRender::createLogicalDevice()
{
  auto const supportedDeviceExtensions = getDeviceExtensions(physicalDevice);
  std::vector<const char*> extensions(deviceExtensions.begin(), deviceExtensions.end());
  bool usePortability = false;

  // Check for timeline semaphore support
  vk::PhysicalDeviceTimelineSemaphoreFeatures timelineSemaphoreFeatures;
  vk::PhysicalDeviceFeatures2 deviceFeatures2;
  deviceFeatures2.pNext = &timelineSemaphoreFeatures;

  physicalDevice.getFeatures2(&deviceFeatures2);

  timelineSemaphoreSupported = timelineSemaphoreFeatures.timelineSemaphore;

  if (timelineSemaphoreSupported && settings::verbose > 1) {
    std::cout << "Timeline semaphores are supported" << std::endl;
  }

  // Add VK_KHR_timeline_semaphore extension if supported
  if (timelineSemaphoreSupported) {
    if (supportedDeviceExtensions.find(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME) != supportedDeviceExtensions.end()) {
      extensions.push_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
      if (settings::verbose > 1) {
        std::cout << "Using timeline semaphore extension" << std::endl;
      }
    } else {
      // If the extension is not available, disable timeline semaphores
      timelineSemaphoreSupported = false;
      if (settings::verbose > 1) {
        std::cout << "Timeline semaphore extension not available, falling back to binary semaphores" << std::endl;
      }
    }
  }

  if (supportedDeviceExtensions.find(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME) != supportedDeviceExtensions.end()) {
    extensions.push_back(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
    if (settings::verbose > 1)
      std::cout << "Using synchronization2 extension" << std::endl;
  }

  if (supportedDeviceExtensions.find(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME) != supportedDeviceExtensions.end()) {
    extensions.push_back(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
    usePortability = true;
  }
  if (View) {
    extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  }
  if (interlock) {
    if (supportedDeviceExtensions.find(VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME) == supportedDeviceExtensions.end()) {
      interlock=false;
    }
    else {
      extensions.emplace_back(VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME);
    }
  }

  if (supportedDeviceExtensions.find(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME) != supportedDeviceExtensions.end()) {
    extensions.push_back(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
    if (settings::verbose > 1)
      std::cout << "Using logical device memory requirements extension"
                << std::endl;
  }

  if (supportedDeviceExtensions.find(VK_KHR_BIND_MEMORY_2_EXTENSION_NAME) != supportedDeviceExtensions.end()) {
    extensions.push_back(VK_KHR_BIND_MEMORY_2_EXTENSION_NAME);
  }

#if defined(DEBUG)
  auto const hasDebugMarkerExt=
    supportedDeviceExtensions.find(VK_EXT_DEBUG_MARKER_EXTENSION_NAME) != supportedDeviceExtensions.end();

  if (hasDebugMarkerExt)
  {
    hasDebugMarker=true;
    extensions.emplace_back(VK_EXT_DEBUG_MARKER_EXTENSION_NAME);
  }
  else
  {
    reportWarning("Debug marker extension not supported");
  }
#endif

  queueFamilyIndices = findQueueFamilies(physicalDevice, View ? &*surface : nullptr);

  std::vector<vk::DeviceQueueCreateInfo> queueCIs;
  std::set<uint32_t> uniqueQueueFamilies = {
    queueFamilyIndices.transferQueueFamily,
    queueFamilyIndices.renderQueueFamily
  };

  if (queueFamilyIndices.presentQueueFamilyFound) {
    uniqueQueueFamilies.emplace(queueFamilyIndices.presentQueueFamily);
  }

  float queuePriority = 1.0f;
  for(auto queueFamily : uniqueQueueFamilies) {
    vk::DeviceQueueCreateInfo queueCI(vk::DeviceQueueCreateFlags(), queueFamily, 1, &queuePriority);
    queueCIs.push_back(queueCI);
  }

  // Build the pNext chain for device features.
  // Start with timeline features if they are supported.
  void * extensionChain = nullptr;
  if (timelineSemaphoreSupported) {
    timelineSemaphoreFeatures.pNext = extensionChain;
    extensionChain = &timelineSemaphoreFeatures;
  }
  auto portabilityFeatures = vk::PhysicalDevicePortabilitySubsetFeaturesKHR(
    false,
    true
  );
  auto interlockFeatures = vk::PhysicalDeviceFragmentShaderInterlockFeaturesEXT(
    true,
    true,
    false
  );
  auto resolveExtension = vk::PhysicalDeviceDepthStencilResolveProperties(
    vk::ResolveModeFlagBits::eMin,
    vk::ResolveModeFlagBits::eMin
  );
  auto props = vk::PhysicalDeviceProperties2(
    {},
    &resolveExtension
  );

  vk::PhysicalDeviceFeatures deviceFeatures;
  deviceFeatures.fillModeNonSolid = true;
  // Needed for some Mac machines.
  deviceFeatures.fragmentStoresAndAtomics = true;
//  deviceFeatures.shaderStorageImageWriteWithoutFormat=true;
//  deviceFeatures.shaderStorageImageReadWithoutFormat=true;

  physicalDevice.getProperties2(&props);

  if (usePortability) {
    portabilityFeatures.pNext = extensionChain;
    extensionChain = &portabilityFeatures;
  }

  if (interlock) {
    interlockFeatures.pNext = extensionChain;
    extensionChain = &interlockFeatures;
  }

  auto deviceCI = vk::DeviceCreateInfo(
    vk::DeviceCreateFlags(),
    VEC_VIEW(queueCIs),
    VEC_VIEW(validationLayers),
    VEC_VIEW(extensions),
    &deviceFeatures,
    extensionChain
  );

  device = physicalDevice.createDeviceUnique(deviceCI, nullptr);
  VULKAN_HPP_DEFAULT_DISPATCHER.init(*device);

  transferQueue = device->getQueue(queueFamilyIndices.transferQueueFamily, 0);
  renderQueue = device->getQueue(queueFamilyIndices.renderQueueFamily, 0);
  if (queueFamilyIndices.presentQueueFamilyFound) {
    presentQueue = device->getQueue(queueFamilyIndices.presentQueueFamily, 0);
  }
}

vk::UniqueSemaphore AsyVkRender::createTimelineSemaphore(uint64_t initialValue) {
  if (!timelineSemaphoreSupported) {
    return device->createSemaphoreUnique({});
  }

  // Create the timeline semaphore type info
  vk::SemaphoreTypeCreateInfo timelineCreateInfo(
    vk::SemaphoreType::eTimeline,
    initialValue
  );

  // Create the semaphore with the timeline type
  vk::SemaphoreCreateInfo createInfo({}, &timelineCreateInfo);

  return device->createSemaphoreUnique(createInfo);
}

void AsyVkRender::signalTimelineSemaphore(vk::Semaphore semaphore, uint64_t value) {
  if (!timelineSemaphoreSupported) return;

  vk::SemaphoreSignalInfo signalInfo(
    semaphore,
    value
  );

  device->signalSemaphore(signalInfo);
}

void AsyVkRender::waitForTimelineSemaphore(vk::Semaphore semaphore, uint64_t value, uint64_t timeout) {
  if (!timelineSemaphoreSupported) return;

  vk::SemaphoreWaitInfo waitInfo(
    {},
    1, &semaphore,
    &value
  );

  // Wait for the semaphore with the specified timeout
  vk::Result result = device->waitSemaphores(waitInfo, timeout);

  if (result == vk::Result::eTimeout) {
    cerr << "warning: Timeline semaphore wait timed out after "
         << 1.0e-9*timeout << " seconds" << endl;
    // Force a full device synchronization and reset timeline values
    try {
      device->waitIdle();
      currentTimelineValue = 0;
    } catch (const std::exception& e) {
      cerr << "Error during device waitIdle after timeout: " << e.what() << endl;
    }
  } else if (result != vk::Result::eSuccess) {
     runtimeError("Timeline semaphore wait failed with result "+
                  std::to_string(static_cast<int>(result)));
  }
}

void AsyVkRender::transitionImageLayout(vk::CommandBuffer cmd,
                             vk::Image image,
                             vk::AccessFlags srcAccessMask,
                             vk::AccessFlags dstAccessMask,
                             vk::ImageLayout oldImageLayout,
                             vk::ImageLayout newImageLayout,
                             vk::PipelineStageFlags srcStageMask,
                             vk::PipelineStageFlags dstStageMask,
                             vk::ImageSubresourceRange subresourceRange)
{
  auto barrier = vk::ImageMemoryBarrier(
    srcAccessMask,
    dstAccessMask,
    oldImageLayout,
    newImageLayout,
    VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
    image,
    subresourceRange
  );

  cmd.pipelineBarrier(srcStageMask, dstStageMask, { }, 0, nullptr, 0, nullptr, 1, &barrier);
}

void AsyVkRender::createExportResources()
{
  auto const cmdInfo = vk::CommandBufferAllocateInfo(
    *renderCommandPool,
    vk::CommandBufferLevel::ePrimary,
    1
  );

  exportCommandBuffer = std::move(device->allocateCommandBuffersUnique(cmdInfo)[0]);
  exportFence = device->createFenceUnique(vk::FenceCreateInfo(vk::FenceCreateFlagBits::eSignaled));
}

void AsyVkRender::createSwapChain()
{
  auto const swapDetails = SwapChainDetails(physicalDevice, *surface);
  auto && format = swapDetails.chooseSurfaceFormat();
  auto && extent = swapDetails.chooseExtent(width,height);

  vk::ImageUsageFlags swapchainImgUsageFlags =
          vk::ImageUsageFlagBits::eColorAttachment
          | vk::ImageUsageFlagBits::eTransferSrc;

  if (fxaa)
  {
    swapchainImgUsageFlags |= vk::ImageUsageFlagBits::eTransferDst;
  }

  vk::SwapchainCreateInfoKHR swapchainCI = vk::SwapchainCreateInfoKHR(
    vk::SwapchainCreateFlagsKHR(),
    *surface,
    swapDetails.chooseImageCount(),
    format.format,
    format.colorSpace,
    extent,
    1,
    swapchainImgUsageFlags,
    vk::SharingMode::eExclusive,
    0,
    nullptr,
    swapDetails.capabilities.currentTransform,
    vk::CompositeAlphaFlagBitsKHR::eOpaque,
    swapDetails.choosePresentMode(),
    VK_TRUE,
    nullptr,
    nullptr
  );

  if (*swapChain) {
    swapchainCI.oldSwapchain = *swapChain;
  }

  if (queueFamilyIndices.renderQueueFamily != queueFamilyIndices.presentQueueFamily) {
    static std::array<std::uint32_t, 2> indices
    {
      queueFamilyIndices.renderQueueFamily,
      queueFamilyIndices.presentQueueFamily
    };

    swapchainCI.imageSharingMode = vk::SharingMode::eConcurrent;
    swapchainCI.queueFamilyIndexCount = indices.size();
    swapchainCI.pQueueFamilyIndices= indices.data();
  }

  swapChain = device->createSwapchainKHRUnique(swapchainCI, nullptr);
  backbufferImages = device->getSwapchainImagesKHR(*swapChain);
  backbufferImageFormat = format.format;
  backbufferExtent = extent;

  for(auto & image: backbufferImages) {
    transitionImageLayout(vk::ImageLayout::eUndefined, vk::ImageLayout::ePresentSrcKHR, image);
  }
}

void AsyVkRender::createOffscreenBuffers() {
  backbufferExtent=vk::Extent2D(width, height);

  auto usageBits=vk::ImageUsageFlagBits::eColorAttachment |
    vk::ImageUsageFlagBits::eTransferSrc;

  if(fxaa)
    usageBits=usageBits | vk::ImageUsageFlagBits::eTransferDst;

  defaultBackbufferImg = createImage(
          backbufferExtent.width,
          backbufferExtent.height,
              vk::SampleCountFlagBits::e1, backbufferImageFormat,
          usageBits,
              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  backbufferImages.emplace_back(defaultBackbufferImg.getImage());

  for(auto & image: backbufferImages) {
    transitionImageLayout(vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal, image);
  }
}

void AsyVkRender::createImageViews()
{
  auto const bufferCount= backbufferImages.size();
  backbufferImageViews.clear();
  backbufferImageViews.reserve(bufferCount);
  for (size_t i= 0; i < bufferCount; ++i)
  {
    vk::ImageViewCreateInfo const viewCI(
            vk::ImageViewCreateFlags(),
            backbufferImages[i],
            vk::ImageViewType::e2D,
            backbufferImageFormat,
            vk::ComponentMapping(),
            vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)
    );
    auto const& imgView= backbufferImageViews.emplace_back(device->createImageViewUnique(viewCI, nullptr));

    setDebugObjectName(*imgView, "backbufferImageView" + std::to_string(i));
  }
}


vk::UniqueShaderModule AsyVkRender::createShaderModule(EShLanguage lang, std::string const & filename, std::vector<std::string> const & options)
{
  std::string header = "#version 450\n";

  for (auto const & option: options) {
    header += "#define " + option + "\n";
  }
  string filePath = locatefile(string(filename));
  auto fileContents= readFile(filePath.c_str());
  fileContents.emplace_back(0); // terminate string

  std::vector<char> source(header.begin(), header.end());
  source.insert(source.end(), fileContents.begin(), fileContents.end());

  std::vector<const char*> const shaderSources {source.data()};
  auto const res = getShaderResources();
  auto const compileMessages = EShMessages(EShMsgSpvRules | EShMsgVulkanRules);
  auto shader = glslang::TShader(lang);
  glslang::TProgram program;
  std::vector<std::uint32_t> spirv;

  shader.setStrings(shaderSources.data(), shaderSources.size());

  if (!shader.parse(&res, 100, false, compileMessages)) {
    std::stringstream s(fileContents.data());
    std::string line;
    unsigned int k=0;
    while(getline(s,line))
      cerr << ++k << ": " << line << std::endl;
    runtimeError("\n failed to parse "
                             + filename
                             + ":\n" + shader.getInfoLog()
                             + " " + shader.getInfoDebugLog());
  }

  program.addShader(&shader);

  if (!program.link(compileMessages)) {
    runtimeError("failed to link shader "
                             + filename
                             + ": " + shader.getInfoLog());
  }

  glslang::GlslangToSpv(*program.getIntermediate(lang), spirv);

  auto shaderModuleCI = vk::ShaderModuleCreateInfo(
    {},
    spirv.size() * sizeof(std::uint32_t),
    spirv.data()
  );

  return device->createShaderModuleUnique(shaderModuleCI);
}

void AsyVkRender::createFramebuffers()
{
  depthFramebuffers.resize(backbufferImageViews.size());
  opaqueGraphicsFramebuffers.resize(backbufferImageViews.size());
  graphicsFramebuffers.resize(backbufferImageViews.size());

  for (auto i= 0u; i < backbufferImageViews.size(); i++)
  {
    // If we are in FXAA, render to an immediate frame buffer
    // to be processed by the fxaa compute shader,
    // otherwise,
    // render directly into swap chain backbuffer
    // still, we should really be moving to scene graphs.
    // The code will get more complicated as times go on
    // (what about multiple post-processing stages, multiple shaders, shadow maps, etc?)

    vk::ImageView const finalRenderTarget =
            fxaa ? *immRenderTargetViews[i]
                 : *backbufferImageViews[i];

    std::array<vk::ImageView, 3> attachments= {*colorImageView, *depthImageView, finalRenderTarget};

    auto depthFramebufferCI = vk::FramebufferCreateInfo(
      {},
      *countRenderPass,
      0, nullptr, backbufferExtent.width, backbufferExtent.height,
      1
    );
    auto opaqueGraphicsFramebufferCI = vk::FramebufferCreateInfo(
      vk::FramebufferCreateFlags(),
      *opaqueGraphicsRenderPass,
      STD_ARR_VIEW(attachments),
      backbufferExtent.width,
      backbufferExtent.height,
      1
    );
    auto graphicsFramebufferCI = vk::FramebufferCreateInfo(
      vk::FramebufferCreateFlags(),
      *graphicsRenderPass,
      STD_ARR_VIEW(attachments),
      backbufferExtent.width,
      backbufferExtent.height,
      1
    );

    depthFramebuffers[i]= device->createFramebufferUnique(depthFramebufferCI);
    opaqueGraphicsFramebuffers[i]= device->createFramebufferUnique(opaqueGraphicsFramebufferCI);
    graphicsFramebuffers[i]= device->createFramebufferUnique(graphicsFramebufferCI);

    setDebugObjectName(*depthFramebuffers[i], "depthFrameBuffer" + std::to_string(i));
    setDebugObjectName(*opaqueGraphicsFramebuffers[i], "opaqueGraphicsFramebuffers" + std::to_string(i));
    setDebugObjectName(*graphicsFramebuffers[i], "graphicsFramebuffers" + std::to_string(i));
  }
}

void AsyVkRender::createCommandPools()
{
  auto transferPoolCI = vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlagBits::eResetCommandBuffer, queueFamilyIndices.transferQueueFamily);
  transferCommandPool = device->createCommandPoolUnique(transferPoolCI);
  auto renderPoolCI = vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlagBits::eResetCommandBuffer, queueFamilyIndices.renderQueueFamily);
  renderCommandPool = device->createCommandPoolUnique(renderPoolCI);
}

void AsyVkRender::createCommandBuffers()
{
  auto renderAllocInfo = vk::CommandBufferAllocateInfo(*renderCommandPool, vk::CommandBufferLevel::ePrimary, static_cast<uint32_t>(maxFramesInFlight * 4));
  auto transferAllocInfo = vk::CommandBufferAllocateInfo(*transferCommandPool, vk::CommandBufferLevel::ePrimary, static_cast<uint32_t>(maxFramesInFlight));
  auto renderCommands = device->allocateCommandBuffersUnique(renderAllocInfo);
  auto transferCommands = device->allocateCommandBuffersUnique(transferAllocInfo);

  for (int i = 0; i < maxFramesInFlight; i++)
  {
    frameObjects[i].commandBuffer = std::move(renderCommands[4 * i]);
    frameObjects[i].countCommandBuffer = std::move(renderCommands[4 * i + 1]);
    frameObjects[i].computeCommandBuffer = std::move(renderCommands[4 * i + 2]);
    frameObjects[i].partialSumsCommandBuffer = std::move(renderCommands[4 * i + 3]);
    frameObjects[i].copyCountCommandBuffer = std::move(transferCommands[i]);
  }
}

vk::CommandBuffer AsyVkRender::beginSingleCommands()
{
  auto const info = vk::CommandBufferAllocateInfo(
    *renderCommandPool,
    vk::CommandBufferLevel::ePrimary,
    1,
    nullptr
  );

  auto const cmd = device->allocateCommandBuffers(info)[0];

  cmd.begin(vk::CommandBufferBeginInfo(
    vk::CommandBufferUsageFlagBits::eOneTimeSubmit
  ));

  return cmd;
}

void AsyVkRender::endSingleCommands(vk::CommandBuffer cmd)
{
  vk::UniqueFence fence = device->createFenceUnique(vk::FenceCreateInfo());

  if (!fence.get()) {
    std::cout << "Fence failed to allocate" << std::endl;
  }

  cmd.end();

  auto info = vk::SubmitInfo();

  info.commandBufferCount = 1;
  info.pCommandBuffers = &cmd;

  vkutils::checkVkResult(renderQueue.submit(1, &info, *fence)); // todo transfer queue
  vkutils::checkVkResult(device->waitForFences(
    1, &*fence, true, std::numeric_limits<std::uint64_t>::max()
  ));

  device->freeCommandBuffers(*renderCommandPool, 1, &cmd);
}

void AsyVkRender::createSyncObjects()
{
  for (auto i = 0; i < maxFramesInFlight; i++) {
    frameObjects[i].imageAvailableSemaphore = device->createSemaphoreUnique(vk::SemaphoreCreateInfo());

    // Create the timeline semaphore for rendering if supported
    renderTimelineSemaphore = createTimelineSemaphore(0);
    frameObjects[i].inCountBufferCopy = device->createSemaphoreUnique(vk::SemaphoreCreateInfo());
    frameObjects[i].inFlightFence = device->createFenceUnique(vk::FenceCreateInfo(vk::FenceCreateFlagBits::eSignaled));
    frameObjects[i].inComputeFence = device->createFenceUnique(vk::FenceCreateInfo(vk::FenceCreateFlagBits::eSignaled));
    frameObjects[i].compressionFinishedEvent = device->createEventUnique(vk::EventCreateInfo());
    frameObjects[i].sumFinishedEvent = device->createEventUnique(vk::EventCreateInfo());
    frameObjects[i].startTimedSumsEvent = device->createEventUnique(vk::EventCreateInfo());
    frameObjects[i].timedSumsFinishedEvent = device->createEventUnique(vk::EventCreateInfo());
    frameObjects[i].renderFinishedSemaphore = device->createSemaphoreUnique(vk::SemaphoreCreateInfo());
  }
}

void AsyVkRender::waitForEvent(vk::Event event) {
  vk::Result result;

  do
  {
    result = device->getEventStatus(event);
  } while(result != vk::Result::eEventSet);
}

uint32_t AsyVkRender::selectMemory(const vk::MemoryRequirements memRequirements, const vk::MemoryPropertyFlags properties)
{
  auto memProperties = physicalDevice.getMemoryProperties();
  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
    if ((memRequirements.memoryTypeBits & (1u << i)) &&
        ((properties & memProperties.memoryTypes[i].propertyFlags) == properties))
      return i;
  runtimeError("failed to find suitable memory type");
  exit(-1);
}

vma::cxx::UniqueBuffer AsyVkRender::createBufferUnique(
        vk::BufferUsageFlags const& usage,
        VkMemoryPropertyFlags const& properties,
        vk::DeviceSize const& size,
        VmaAllocationCreateFlags const& vmaFlags,
        VmaMemoryUsage const& memoryUsage,
        const char * bufferName)
{
  auto bufferCI = vk::BufferCreateInfo(vk::BufferCreateFlags(), size, usage);

  VmaAllocationCreateInfo createInfo = {};
  createInfo.usage = memoryUsage;
  createInfo.requiredFlags = properties;
  createInfo.flags=vmaFlags;

  if (bufferName != nullptr && settings::verbose > 2) {
    std::cout << "Creating buffer " << bufferName << " of size: " << size << std::endl;
  }

  return allocator.createBuffer(bufferCI, createInfo);
}

void AsyVkRender::copyBufferToBuffer(const vk::Buffer& srcBuffer, const vk::Buffer& dstBuffer, const vk::DeviceSize size)
{
  auto allocInfo = vk::CommandBufferAllocateInfo(*transferCommandPool, vk::CommandBufferLevel::ePrimary, 1);
  auto commandBuffer = std::move(device->allocateCommandBuffersUnique(allocInfo)[0]);

  auto commandBufferBeginInfo = vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
  commandBuffer->begin(commandBufferBeginInfo);
  auto copyRegion = vk::BufferCopy(0, 0, size);
  commandBuffer->copyBuffer(srcBuffer, dstBuffer, copyRegion);
  commandBuffer->end();

  auto fence = device->createFenceUnique(vk::FenceCreateInfo());
  auto submitInfo = vk::SubmitInfo(0, nullptr, nullptr, 1, &*commandBuffer);
  auto submitResult = transferQueue.submit(1, &submitInfo, *fence);
  if (submitResult != vk::Result::eSuccess)
    runtimeError("failed to submit command buffer");
  vkutils::checkVkResult(device->waitForFences(
    1, &*fence, VK_TRUE, timeout
  ));
}

void AsyVkRender::copyToBuffer(
        const vk::Buffer& buffer,
        const void* data,
        vk::DeviceSize size,
        vma::cxx::UniqueBuffer const& stagingBuffer
        )
{
  if (false) {
    auto externalMemoryBufferCI = vk::ExternalMemoryBufferCreateInfo(vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT);
    auto bufferCI = vk::BufferCreateInfo(vk::BufferCreateFlags(), size, vk::BufferUsageFlagBits::eTransferSrc, vk::SharingMode::eExclusive, 0, nullptr, &externalMemoryBufferCI);
    auto hostBuffer = device->createBufferUnique(bufferCI);

    copyBufferToBuffer(*hostBuffer, buffer, size);
  } else {
    vma::cxx::MemoryMapperLock const stgBufMemPtr(stagingBuffer);
    memcpy(stgBufMemPtr.getCopyPtr(), data, size);
    copyBufferToBuffer(stagingBuffer.getBuffer(), buffer, size);
  }
}

void AsyVkRender::setDebugObjectName(
        uint64_t const& object,
        vk::DebugReportObjectTypeEXT const& objType,
        std::string const& name
        )
{
#if defined(DEBUG)
  if (hasDebugMarker)
  {
    vk::DebugMarkerObjectNameInfoEXT const tagInfo(objType, object, name.c_str());
    device->debugMarkerSetObjectNameEXT(tagInfo);
  }
#endif
}

void AsyVkRender::copyToBuffer(
        const vk::Buffer& buffer,
        const void* data,
        vk::DeviceSize size
)
{
  vma::cxx::UniqueBuffer copyToStageBf = createBufferUnique(
          vk::BufferUsageFlagBits::eTransferSrc,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          size,
          VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
          VMA_MEMORY_USAGE_AUTO,
          VARIABLE_NAME(copyToStageBf)
          );

  copyToBuffer(buffer, data, size, copyToStageBf);
}

void AsyVkRender::zeroBuffer(vk::CommandBuffer const& cmdBuffer,
                             vk::Buffer const& buffer)
{
  cmdBuffer.fillBuffer(buffer, 0, vk::WholeSize, 0);
}

vma::cxx::UniqueImage AsyVkRender::createImage(
        std::uint32_t w, std::uint32_t h, vk::SampleCountFlagBits samples, vk::Format fmt, vk::ImageUsageFlags usage,
        VkMemoryPropertyFlags props, vk::ImageType type, std::uint32_t depth
)
{
  auto info = vk::ImageCreateInfo();

  info.imageType      = type;
  info.extent         = vk::Extent3D(w, h, depth);
  info.mipLevels      = 1;
  info.arrayLayers    = 1;
  info.format         = fmt;
  info.tiling         = vk::ImageTiling::eOptimal;
  info.initialLayout  = vk::ImageLayout::eUndefined;
  info.usage          = usage;
  info.sharingMode    = vk::SharingMode::eExclusive;
  info.samples        = samples;

  VmaAllocationCreateInfo allocCreateInfo = {};
  allocCreateInfo.requiredFlags= props;
  allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;


  return allocator.createImage(info, allocCreateInfo);
}

void AsyVkRender::createImageView(vk::Format fmt, vk::ImageAspectFlagBits flags,
                                  vk::Image const& img, vk::UniqueImageView& imgView,
                                  vk::ImageViewType type)
{
  auto info = vk::ImageViewCreateInfo();

  info.image = img;
  info.viewType = type;
  info.format = fmt;
  info.components = vk::ComponentMapping();
  info.subresourceRange = vk::ImageSubresourceRange(
    flags,
    0,
    1,
    0,
    1
  );

  imgView = device->createImageViewUnique(info);
}

void AsyVkRender::copyFromBuffer(const vk::Buffer& buffer, void* data, vk::DeviceSize size)
{
  vma::cxx::UniqueBuffer copyFromStageBf= createBufferUnique(
          vk::BufferUsageFlagBits::eTransferDst,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          size,
          VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
          VMA_MEMORY_USAGE_AUTO,
          VARIABLE_NAME(copyFromStageBf)
  );

  auto const cmd = beginSingleCommands();
  auto const cpy = vk::BufferCopy(
    0, 0, size
  );

  cmd.copyBuffer(buffer, copyFromStageBf.getBuffer(), 1, &cpy);

  endSingleCommands(cmd);

  vma::cxx::MemoryMapperLock const mappedMem(copyFromStageBf);
  memcpy(data, mappedMem.getCopyPtr(), size);
}

void AsyVkRender::createImageSampler(vk::UniqueSampler & sampler) {

  auto info = vk::SamplerCreateInfo(
    vk::SamplerCreateFlags(),
    vk::Filter::eLinear,
    vk::Filter::eLinear,
    vk::SamplerMipmapMode::eNearest,
    vk::SamplerAddressMode::eRepeat,
    vk::SamplerAddressMode::eClampToEdge,
    vk::SamplerAddressMode::eClampToEdge,
    0.f,
    false,
    0.f,
    false,
    vk::CompareOp::eAlways,
    0.f,
    0.f
  );

  sampler = device->createSamplerUnique(info);
}

void AsyVkRender::transitionImageLayout(vk::ImageLayout from, vk::ImageLayout to, vk::Image img) {

  auto const cmd = beginSingleCommands();
  auto barrier = vk::ImageMemoryBarrier(
    vk::AccessFlagBits::eMemoryWrite,
    vk::AccessFlagBits::eMemoryWrite,
    from,
    to,
    VK_QUEUE_FAMILY_IGNORED,
    VK_QUEUE_FAMILY_IGNORED,
    img
  );

  barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;
  cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                      vk::PipelineStageFlagBits::eTransfer,
                      {},
                      0,
                      nullptr,
                      0,
                      nullptr,
                      1,
                      &barrier);
  endSingleCommands(cmd);
}

void AsyVkRender::copyDataToImage(const void *data, vk::DeviceSize size, vk::Image img,
                                  std::uint32_t w, std::uint32_t h, vk::Offset3D const & offset) {

  vma::cxx::UniqueBuffer copyToImageStageBf = createBufferUnique(
          vk::BufferUsageFlagBits::eTransferSrc,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          size,
          VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
          VMA_MEMORY_USAGE_AUTO,
          VARIABLE_NAME(copyToImageStageBf)
    );

  if (vma::cxx::MemoryMapperLock mappedMem(copyToImageStageBf); true)
  {
    memcpy(mappedMem.getCopyPtr<uint8_t>(), data, size);
  }

  auto const cmd = beginSingleCommands();
  auto cpy = vk::BufferImageCopy(
    0,
    0,
    0
  );

  cpy.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
  cpy.imageSubresource.mipLevel = 0;
  cpy.imageSubresource.baseArrayLayer = 0;
  cpy.imageSubresource.layerCount = 1;
  cpy.imageOffset = offset;
  cpy.imageExtent = vk::Extent3D {
      w,
      h,
      1
  };

  cmd.copyBufferToImage(copyToImageStageBf.getBuffer(), img, vk::ImageLayout::eTransferDstOptimal, 1, &cpy);

  endSingleCommands(cmd);
}

void AsyVkRender::setDeviceBufferData(DeviceBuffer& buffer, const void* data, vk::DeviceSize size, size_t nobjects)
{
  // Vulkan doesn't allow a buffer to have a size of 0
  vk::BufferCreateInfo(vk::BufferCreateFlags(), std::max(vk::DeviceSize(16), size), buffer.usage);
  buffer._buffer = createBufferUnique(
                          buffer.usage,
                          buffer.properties,
                          size
                          );

  buffer.nobjects = nobjects;
  if (size > buffer.stgBufferSize) {
    // minimum array size of 16 bytes to avoid some Vulkan issues
    vk::DeviceSize newSize = 16;
    while (newSize < size) newSize *= 2;
    buffer.stgBufferSize = newSize;

    // check whether we need a staging buffer
    if (true) {
      buffer._stgBuffer = createBufferUnique(
              vk::BufferUsageFlagBits::eTransferSrc,
              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
              buffer.stgBufferSize,
              VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
      );
    }
  }

  if (data) {
    if (false) {
      copyToBuffer(buffer._buffer.getBuffer(), data, size);
    } else {
      copyToBuffer(buffer._buffer.getBuffer(), data, size, buffer._stgBuffer);
    }
  }
}

void AsyVkRender::createDescriptorSetLayout()
{
  auto uboLayoutBinding = vk::DescriptorSetLayoutBinding(
    0,
    vk::DescriptorType::eUniformBuffer,
    1,
    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment
  );
  auto materialBufferBinding = vk::DescriptorSetLayoutBinding(
    1,
    vk::DescriptorType::eStorageBuffer,
    1,
    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment
  );
  auto lightBufferBinding = vk::DescriptorSetLayoutBinding(
    2,
    vk::DescriptorType::eStorageBuffer,
    1,
    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment
  );
  auto countBufferBinding = vk::DescriptorSetLayoutBinding(
    3,
    vk::DescriptorType::eStorageBuffer,
    1,
    vk::ShaderStageFlagBits::eFragment
  );
  auto offsetBufferBinding = vk::DescriptorSetLayoutBinding(
    4,
    vk::DescriptorType::eStorageBuffer,
    1,
    vk::ShaderStageFlagBits::eFragment
  );
  auto fragmentBufferBinding = vk::DescriptorSetLayoutBinding(
    5,
    vk::DescriptorType::eStorageBuffer,
    1,
    vk::ShaderStageFlagBits::eFragment
  );
  auto depthBufferBinding = vk::DescriptorSetLayoutBinding(
    6,
    vk::DescriptorType::eStorageBuffer,
    1,
    vk::ShaderStageFlagBits::eFragment
  );
  auto opaqueBufferBinding = vk::DescriptorSetLayoutBinding(
    7,
    vk::DescriptorType::eStorageBuffer,
    1,
    vk::ShaderStageFlagBits::eFragment
  );
  auto opaqueDepthBufferBinding = vk::DescriptorSetLayoutBinding(
    8,
    vk::DescriptorType::eStorageBuffer,
    1,
    vk::ShaderStageFlagBits::eFragment
  );
  auto indexBufferBinding = vk::DescriptorSetLayoutBinding(
    9,
    vk::DescriptorType::eStorageBuffer,
    1,
    vk::ShaderStageFlagBits::eFragment
  );
  auto elementBufferBinding = vk::DescriptorSetLayoutBinding(
    10,
    vk::DescriptorType::eStorageBuffer,
    1,
    vk::ShaderStageFlagBits::eFragment
  );
  auto irradianceSamplerBinding = vk::DescriptorSetLayoutBinding(
    11,
    vk::DescriptorType::eCombinedImageSampler,
    1,
    vk::ShaderStageFlagBits::eFragment
  );
  auto brdfSamplerBinding = vk::DescriptorSetLayoutBinding(
    12,
    vk::DescriptorType::eCombinedImageSampler,
    1,
    vk::ShaderStageFlagBits::eFragment
  );
  auto reflectionSamplerBinding = vk::DescriptorSetLayoutBinding(
    13,
    vk::DescriptorType::eCombinedImageSampler,
    1,
    vk::ShaderStageFlagBits::eFragment
  );

  std::vector<vk::DescriptorSetLayoutBinding> layoutBindings {
    uboLayoutBinding,
    materialBufferBinding,
    lightBufferBinding,
    countBufferBinding,
    offsetBufferBinding,
    fragmentBufferBinding,
    depthBufferBinding,
    opaqueBufferBinding,
    opaqueDepthBufferBinding,
    indexBufferBinding,
    elementBufferBinding
  };

  if (ibl) {
    layoutBindings.emplace_back(irradianceSamplerBinding);
    layoutBindings.emplace_back(brdfSamplerBinding);
    layoutBindings.emplace_back(reflectionSamplerBinding);
  }

  auto layoutCI = vk::DescriptorSetLayoutCreateInfo(
    vk::DescriptorSetLayoutCreateFlags(),
    layoutBindings.size(),
    &layoutBindings[0]
  );
  materialDescriptorSetLayout = device->createDescriptorSetLayoutUnique(layoutCI);
}

void AsyVkRender::createComputeDescriptorSetLayout()
{
  // gpu indexing
  std::vector<vk::DescriptorSetLayoutBinding> layoutBindings
  {
    vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute),
    vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute),
    vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute),
    vk::DescriptorSetLayoutBinding(3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute)
  };
  auto layoutCI = vk::DescriptorSetLayoutCreateInfo(
    vk::DescriptorSetLayoutCreateFlags(),
    layoutBindings.size(),
    &layoutBindings[0]
  );

  computeDescriptorSetLayout = device->createDescriptorSetLayoutUnique(layoutCI);

  // post processing

  if (fxaa)
  {
    std::vector<vk::DescriptorSetLayoutBinding> const postProcessingLayoutBindings{
            {0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eCompute},
            {1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute},
            {2, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute},
    };

    postProcessDescSetLayout= device->createDescriptorSetLayoutUnique({{}, VEC_VIEW(postProcessingLayoutBindings)});
  }
}

void AsyVkRender::createDescriptorPool()
{
  std::vector<vk::DescriptorPoolSize> poolSizes;

  poolSizes.resize(11);
  poolSizes[0].type = vk::DescriptorType::eUniformBuffer;
  poolSizes[0].descriptorCount = maxFramesInFlight;

  poolSizes[1].type = vk::DescriptorType::eStorageBuffer;
  poolSizes[1].descriptorCount = maxFramesInFlight;

  poolSizes[2].type = vk::DescriptorType::eStorageBuffer;
  poolSizes[2].descriptorCount = maxFramesInFlight;

  poolSizes[3].type = vk::DescriptorType::eStorageBuffer;
  poolSizes[3].descriptorCount = maxFramesInFlight;

  poolSizes[4].type = vk::DescriptorType::eStorageBuffer;
  poolSizes[4].descriptorCount = maxFramesInFlight;

  poolSizes[5].type = vk::DescriptorType::eStorageBuffer;
  poolSizes[5].descriptorCount = maxFramesInFlight;

  poolSizes[6].type = vk::DescriptorType::eStorageBuffer;
  poolSizes[6].descriptorCount = maxFramesInFlight;

  poolSizes[7].type = vk::DescriptorType::eStorageBuffer;
  poolSizes[7].descriptorCount = maxFramesInFlight;

  poolSizes[8].type = vk::DescriptorType::eStorageBuffer;
  poolSizes[8].descriptorCount = maxFramesInFlight;

  poolSizes[9].type = vk::DescriptorType::eStorageBuffer;
  poolSizes[9].descriptorCount = maxFramesInFlight;

  poolSizes[10].type = vk::DescriptorType::eStorageBuffer;
  poolSizes[10].descriptorCount = maxFramesInFlight;

  if (ibl) {
    poolSizes.emplace_back(
      vk::DescriptorPoolSize(
        vk::DescriptorType::eCombinedImageSampler,
        maxFramesInFlight
      )
    );
    poolSizes.emplace_back(
      vk::DescriptorPoolSize(
        vk::DescriptorType::eCombinedImageSampler,
        maxFramesInFlight
      )
    );
    poolSizes.emplace_back(
      vk::DescriptorPoolSize(
        vk::DescriptorType::eCombinedImageSampler,
        maxFramesInFlight
      )
    );
  }

  auto poolCI = vk::DescriptorPoolCreateInfo(
    vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
    maxFramesInFlight,
    poolSizes.size(),
    &poolSizes[0]
  );
  descriptorPool = device->createDescriptorPoolUnique(poolCI);
}

void AsyVkRender::createComputeDescriptorPool()
{
  // gpu indexing

  std::array<vk::DescriptorPoolSize, 4> poolSizes;

  // countBuffer
  poolSizes[0].type = vk::DescriptorType::eStorageBuffer;
  poolSizes[0].descriptorCount = 1;

  // globalSumBuffer
  poolSizes[1].type = vk::DescriptorType::eStorageBuffer;
  poolSizes[1].descriptorCount = 1;

  // offsetBuffer
  poolSizes[2].type = vk::DescriptorType::eStorageBuffer;
  poolSizes[2].descriptorCount = 1;

  // feedbackBuffer
  poolSizes[3].type = vk::DescriptorType::eStorageBuffer;
  poolSizes[3].descriptorCount = 1;

  auto poolCI = vk::DescriptorPoolCreateInfo(
    vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
    1,
    poolSizes.size(),
    &poolSizes[0]
  );
  computeDescriptorPool = device->createDescriptorPoolUnique(poolCI);

  // post processing

  if (fxaa)
  {
    auto const poolSetCount= static_cast<uint32_t>(backbufferImages.size());

    std::vector<vk::DescriptorPoolSize> const postProcPoolSizes{
            {vk::DescriptorType::eCombinedImageSampler, poolSetCount},// input image
            {vk::DescriptorType::eStorageImage, poolSetCount},        // input image, non-sampled
            {vk::DescriptorType::eStorageImage, poolSetCount},        // output image image
    };

    postProcessDescPool= device->createDescriptorPoolUnique(
            {vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, poolSetCount, VEC_VIEW(postProcPoolSizes)}
    );
  }
}

void AsyVkRender::createDescriptorSets()
{
  std::vector<vk::DescriptorSetLayout> layouts(maxFramesInFlight, *materialDescriptorSetLayout);
  auto allocInfo = vk::DescriptorSetAllocateInfo(
    *descriptorPool,
    VEC_VIEW(layouts)
  );
  auto descriptorSets = device->allocateDescriptorSetsUnique(allocInfo);

  for (auto i = 0; i < maxFramesInFlight; i++)
    frameObjects[i].descriptorSet = std::move(descriptorSets[i]);

  auto computeAllocInfo = vk::DescriptorSetAllocateInfo(
    *computeDescriptorPool,
    1,
    &*computeDescriptorSetLayout
  );

  computeDescriptorSet = std::move(device->allocateDescriptorSetsUnique(computeAllocInfo)[0]);

  // post processing descs

  if (fxaa)
  {
    std::vector postProcessDescLayouts(backbufferImages.size(), *postProcessDescSetLayout);
    postProcessDescSet= device->allocateDescriptorSetsUnique({*postProcessDescPool, VEC_VIEW(postProcessDescLayouts)});
  }
}

void AsyVkRender::writeDescriptorSets(bool transparent)
{
  for (auto i = 0; i < maxFramesInFlight; i++)
  {
    auto uboInfo = vk::DescriptorBufferInfo();

    uboInfo.buffer = frameObjects[i].uboBf.getBuffer();
    uboInfo.offset = 0;
    uboInfo.range = sizeof(UniformBufferObject);

    auto countBufferInfo = vk::DescriptorBufferInfo();
    auto offsetBufferInfo = vk::DescriptorBufferInfo();
    auto opaqueBufferInfo = vk::DescriptorBufferInfo();
    auto opaqueDepthBufferInfo = vk::DescriptorBufferInfo();
    auto indexBufferInfo = vk::DescriptorBufferInfo();
    auto elementBufferInfo = vk::DescriptorBufferInfo();

    if(transparent) {
      countBufferInfo.buffer = countBf.getBuffer();
      countBufferInfo.offset = 0;
      countBufferInfo.range = countBufferSize;

      offsetBufferInfo.buffer = offsetBf.getBuffer();
      offsetBufferInfo.offset = 0;
      offsetBufferInfo.range = offsetBufferSize;

      opaqueBufferInfo.buffer = opaqueBf.getBuffer();
      opaqueBufferInfo.offset = 0;
      opaqueBufferInfo.range = opaqueBufferSize;

      opaqueDepthBufferInfo.buffer = opaqueDepthBf.getBuffer();
      opaqueDepthBufferInfo.offset = 0;
      opaqueDepthBufferInfo.range = opaqueDepthBufferSize;

      indexBufferInfo.buffer = indexBf.getBuffer();
      indexBufferInfo.offset = 0;
      indexBufferInfo.range = indexBufferSize;

      elementBufferInfo.buffer = elementBf.getBuffer();
      elementBufferInfo.offset = 0;
      elementBufferInfo.range = elementBufferSize;
    }

    std::array<vk::WriteDescriptorSet,7> writes;

    writes[0].dstSet = *frameObjects[i].descriptorSet;
    writes[0].dstBinding = 0;
    writes[0].dstArrayElement = 0;
    writes[0].descriptorType = vk::DescriptorType::eUniformBuffer;
    writes[0].descriptorCount = 1;
    writes[0].pBufferInfo = &uboInfo;

    if(transparent) {
      writes[1].dstSet = *frameObjects[i].descriptorSet;
      writes[1].dstBinding = 3;
      writes[1].dstArrayElement = 0;
      writes[1].descriptorType = vk::DescriptorType::eStorageBuffer;
      writes[1].descriptorCount = 1;
      writes[1].pBufferInfo = &countBufferInfo;

      writes[2].dstSet = *frameObjects[i].descriptorSet;
      writes[2].dstBinding = 4;
      writes[2].dstArrayElement = 0;
      writes[2].descriptorType = vk::DescriptorType::eStorageBuffer;
      writes[2].descriptorCount = 1;
      writes[2].pBufferInfo = &offsetBufferInfo;

      writes[3].dstSet = *frameObjects[i].descriptorSet;
      writes[3].dstBinding = 7;
      writes[3].dstArrayElement = 0;
      writes[3].descriptorType = vk::DescriptorType::eStorageBuffer;
      writes[3].descriptorCount = 1;
      writes[3].pBufferInfo = &opaqueBufferInfo;

      writes[4].dstSet = *frameObjects[i].descriptorSet;
      writes[4].dstBinding = 8;
      writes[4].dstArrayElement = 0;
      writes[4].descriptorType = vk::DescriptorType::eStorageBuffer;
      writes[4].descriptorCount = 1;
      writes[4].pBufferInfo = &opaqueDepthBufferInfo;

      if(GPUcompress) {
        writes[5].dstSet = *frameObjects[i].descriptorSet;
        writes[5].dstBinding = 9;
        writes[5].dstArrayElement = 0;
        writes[5].descriptorType = vk::DescriptorType::eStorageBuffer;
        writes[5].descriptorCount = 1;
        writes[5].pBufferInfo = &indexBufferInfo;

        writes[6].dstSet = *frameObjects[i].descriptorSet;
        writes[6].dstBinding = 10;
        writes[6].dstArrayElement = 0;
        writes[6].descriptorType = vk::DescriptorType::eStorageBuffer;
        writes[6].descriptorCount = 1;
        writes[6].pBufferInfo = &elementBufferInfo;
      }
    }

    device->updateDescriptorSets(transparent ? (GPUcompress ? 7 : 5) : 1,
                                 writes.data(),0,nullptr);

    if (ibl) {
      auto irradianceSampInfo = vk::DescriptorImageInfo();

      irradianceSampInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
      irradianceSampInfo.imageView = *irradianceView;
      irradianceSampInfo.sampler = *irradianceSampler;

      auto brdfSampInfo = vk::DescriptorImageInfo();

      brdfSampInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
      brdfSampInfo.imageView = *brdfView;
      brdfSampInfo.sampler = *brdfSampler;

      auto reflSampInfo = vk::DescriptorImageInfo();

      reflSampInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
      reflSampInfo.imageView = *reflectionView;
      reflSampInfo.sampler = *reflectionSampler;

      std::array<vk::WriteDescriptorSet, 3> samplerWrites;

      samplerWrites[0].dstSet = *frameObjects[i].descriptorSet;
      samplerWrites[0].dstBinding = 11;
      samplerWrites[0].dstArrayElement = 0;
      samplerWrites[0].descriptorType = vk::DescriptorType::eCombinedImageSampler;
      samplerWrites[0].descriptorCount = 1;
      samplerWrites[0].pImageInfo = &irradianceSampInfo;

      samplerWrites[1].dstSet = *frameObjects[i].descriptorSet;
      samplerWrites[1].dstBinding = 12;
      samplerWrites[1].dstArrayElement = 0;
      samplerWrites[1].descriptorType = vk::DescriptorType::eCombinedImageSampler;
      samplerWrites[1].descriptorCount = 1;
      samplerWrites[1].pImageInfo = &brdfSampInfo;

      samplerWrites[2].dstSet = *frameObjects[i].descriptorSet;
      samplerWrites[2].dstBinding = 13;
      samplerWrites[2].dstArrayElement = 0;
      samplerWrites[2].descriptorType = vk::DescriptorType::eCombinedImageSampler;
      samplerWrites[2].descriptorCount = 1;
      samplerWrites[2].pImageInfo = &reflSampInfo;

      device->updateDescriptorSets(samplerWrites.size(), samplerWrites.data(), 0, nullptr);
    }
  }

  if(transparent) {
    // compute descriptors

    auto countBufferInfo = vk::DescriptorBufferInfo();

    countBufferInfo.buffer = countBf.getBuffer();
    countBufferInfo.offset = 0;
    countBufferInfo.range = countBufferSize;

    auto globalSumBufferInfo = vk::DescriptorBufferInfo();

    globalSumBufferInfo.buffer = globalSumBf.getBuffer();
    globalSumBufferInfo.offset = 0;
    globalSumBufferInfo.range = globalSize;

    auto offsetBufferInfo = vk::DescriptorBufferInfo();

    offsetBufferInfo.buffer = offsetBf.getBuffer();
    offsetBufferInfo.offset = 0;
    offsetBufferInfo.range = offsetBufferSize;

    auto feedbackBufferInfo = vk::DescriptorBufferInfo();

    feedbackBufferInfo.buffer = feedbackBf.getBuffer();
    feedbackBufferInfo.offset = 0;
    feedbackBufferInfo.range = feedbackBufferSize;

    std::array<vk::WriteDescriptorSet, 4> writes;

    writes[0].dstSet = *computeDescriptorSet;
    writes[0].dstBinding = 0;
    writes[0].dstArrayElement = 0;
    writes[0].descriptorType = vk::DescriptorType::eStorageBuffer;
    writes[0].descriptorCount = 1;
    writes[0].pBufferInfo = &countBufferInfo;

    writes[1].dstSet = *computeDescriptorSet;
    writes[1].dstBinding = 1;
    writes[1].dstArrayElement = 0;
    writes[1].descriptorType = vk::DescriptorType::eStorageBuffer;
    writes[1].descriptorCount = 1;
    writes[1].pBufferInfo = &globalSumBufferInfo;

    writes[2].dstSet = *computeDescriptorSet;
    writes[2].dstBinding = 2;
    writes[2].dstArrayElement = 0;
    writes[2].descriptorType = vk::DescriptorType::eStorageBuffer;
    writes[2].descriptorCount = 1;
    writes[2].pBufferInfo = &offsetBufferInfo;

    writes[3].dstSet = *computeDescriptorSet;
    writes[3].dstBinding = 3;
    writes[3].dstArrayElement = 0;
    writes[3].descriptorType = vk::DescriptorType::eStorageBuffer;
    writes[3].descriptorCount = 1;
    writes[3].pBufferInfo = &feedbackBufferInfo;

    device->updateDescriptorSets(writes.size(), writes.data(), 0, nullptr);
  }

  if(fxaa)
    writePostProcessDescSets();
}

void AsyVkRender::writePostProcessDescSets()
{
  // post process descriptors
  for (size_t i=0; i < backbufferImages.size(); ++i)
  {
    vk::DescriptorImageInfo inputImgInfo(
            *immRenderTargetSampler[i],
            *immRenderTargetViews[i],
            vk::ImageLayout::eGeneral
            );
    vk::DescriptorImageInfo inputImgInfoNonSampled(
            {},
            *immRenderTargetViews[i],
            vk::ImageLayout::eGeneral
    );
    vk::DescriptorImageInfo outputImgInfo({}, *prePresentationImgViews[i], vk::ImageLayout::eGeneral);

    std::vector<vk::WriteDescriptorSet> const postProcDescWrite{
            {*postProcessDescSet[i], 0, 0, 1, vk::DescriptorType::eCombinedImageSampler, &inputImgInfo},
            {*postProcessDescSet[i], 1, 0, 1, vk::DescriptorType::eStorageImage, &inputImgInfoNonSampled},
            {*postProcessDescSet[i], 2, 0, 1, vk::DescriptorType::eStorageImage, &outputImgInfo}
    };

    device->updateDescriptorSets(VEC_VIEW(postProcDescWrite), EMPTY_VIEW);
  }
}

void AsyVkRender::writeMaterialAndLightDescriptors() {

  for (auto i = 0; i < maxFramesInFlight; i++) {
    auto materialBufferInfo = vk::DescriptorBufferInfo();

    materialBufferInfo.buffer = materialBf.getBuffer();
    materialBufferInfo.offset = 0;
    materialBufferInfo.range = sizeof(camp::Material) * nmaterials;

    auto lightBufferInfo = vk::DescriptorBufferInfo();

    lightBufferInfo.buffer = lightBf.getBuffer();
    lightBufferInfo.offset = 0;
    lightBufferInfo.range = sizeof(Light) * nlights;

    std::array<vk::WriteDescriptorSet, 2> writes;

    writes[0].dstSet = *frameObjects[i].descriptorSet;
    writes[0].dstBinding = 1;
    writes[0].dstArrayElement = 0;
    writes[0].descriptorType = vk::DescriptorType::eStorageBuffer;
    writes[0].descriptorCount = 1;
    writes[0].pBufferInfo = &materialBufferInfo;

    writes[1].dstSet = *frameObjects[i].descriptorSet;
    writes[1].dstBinding = 2;
    writes[1].dstArrayElement = 0;
    writes[1].descriptorType = vk::DescriptorType::eStorageBuffer;
    writes[1].descriptorCount = 1;
    writes[1].pBufferInfo = &lightBufferInfo;

    device->updateDescriptorSets(writes.size(), writes.data(), 0, nullptr);
  }
}

void AsyVkRender::updateSceneDependentBuffers() {

  fragmentBufferSize = maxFragments*sizeof(vec4);
  fragmentBf = createBufferUnique(
          vk::BufferUsageFlagBits::eStorageBuffer,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
          fragmentBufferSize,
          0,
          VMA_MEMORY_USAGE_AUTO,
          VARIABLE_NAME(fragmentBf));

  depthBufferSize = maxFragments*sizeof(float);
  depthBf = createBufferUnique(
          vk::BufferUsageFlagBits::eStorageBuffer,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
          depthBufferSize,
          0,
          VMA_MEMORY_USAGE_AUTO,
          VARIABLE_NAME(depthBf));

  for(auto i = 0; i < maxFramesInFlight; i++) {

    auto fragmentBufferInfo = vk::DescriptorBufferInfo(
      fragmentBf.getBuffer(),
      0,
      fragmentBufferSize
    );
    auto depthBufferInfo = vk::DescriptorBufferInfo(
      depthBf.getBuffer(),
      0,
      depthBufferSize
    );

    std::array<vk::WriteDescriptorSet, 2> writes;

    writes[0] = vk::WriteDescriptorSet(
      *frameObjects[i].descriptorSet,
      5,
      0,
      1,
      vk::DescriptorType::eStorageBuffer,
      nullptr,
      &fragmentBufferInfo,
      nullptr
    );
    writes[1] = vk::WriteDescriptorSet(
      *frameObjects[i].descriptorSet,
      6,
      0,
      1,
      vk::DescriptorType::eStorageBuffer,
      nullptr,
      &depthBufferInfo,
      nullptr
    );

    // todo remove frame-dependent descriptor sets
    device->updateDescriptorSets(writes.size(), writes.data(), 0, nullptr);
  }

  // if the fragment buffer size changes, all transparent data needs
  // to be copied again to the GPU for every frame in flight.
  transparentData.renderCount = 0;
}

void AsyVkRender::createBuffers()
{
  feedbackBufferSize=2*sizeof(std::uint32_t);
  elementBufferSize=sizeof(std::uint32_t);

  feedbackBf = createBufferUnique(
    vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
    feedbackBufferSize,
    VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
    VMA_MEMORY_USAGE_AUTO,
    VARIABLE_NAME(feedbackBf)
  );

  if(GPUcompress)
  {
    elementBf= createBufferUnique(
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
            elementBufferSize,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
    VMA_MEMORY_USAGE_AUTO,
      VARIABLE_NAME(elementBf)
    );
  }

  for (auto& frameObj : frameObjects)
  {
    frameObj.uboBf = createBufferUnique(
      vk::BufferUsageFlagBits::eUniformBuffer,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      sizeof(UniformBufferObject),
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
      VMA_MEMORY_USAGE_AUTO,
      VARIABLE_NAME(frameObj.uboBf)
    );
    frameObj.uboMappedMemory = make_unique<vma::cxx::MemoryMapperLock>(frameObj.uboBf);
  }

  createMaterialAndLightBuffers();
}


void AsyVkRender::createMaterialAndLightBuffers() {
  if(nmaterials > 0)
    materialBf = createBufferUnique(
      vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      sizeof(camp::Material) * nmaterials,
      0,
      VMA_MEMORY_USAGE_AUTO,
      VARIABLE_NAME(materialBf));

  if(nlights > 0)
    lightBf = createBufferUnique(
      vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      sizeof(camp::Light) * nlights,
      0,
      VMA_MEMORY_USAGE_AUTO,
      VARIABLE_NAME(lightBf));
}

void AsyVkRender::createImmediateRenderTargets()
{
  // Choose post-process format: FXAA requires RGBA8 to match layout rgba8
  postProcFormat = fxaa ? vk::Format::eR8G8B8A8Unorm : backbufferImageFormat;
  immRenderTargetViews.clear();
  immediateRenderTargetImgs.clear();
  prePresentationImages.clear();
  prePresentationImgViews.clear();
  immRenderTargetSampler.clear();

  auto const framebufferSize= backbufferImages.size();

  immRenderTargetViews.reserve(framebufferSize);
  immediateRenderTargetImgs.reserve(framebufferSize);
  prePresentationImages.reserve(framebufferSize);
  prePresentationImgViews.reserve(framebufferSize);
  immRenderTargetSampler.reserve(framebufferSize);

  for (size_t i= 0; i < framebufferSize; ++i)
  {
    // for immediate render target (after pixel shader)
    auto const& immRenderTarget= immediateRenderTargetImgs.emplace_back(createImage(
            backbufferExtent.width,
            backbufferExtent.height,
            vk::SampleCountFlagBits::e1,
            postProcFormat,
            vk::ImageUsageFlagBits::eColorAttachment
                    | vk::ImageUsageFlagBits::eSampled
                    | vk::ImageUsageFlagBits::eStorage,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    ));

    setDebugObjectName(vk::Image(immRenderTarget.getImage()), "immediateRenderTargetImg" + std::to_string(i));

    auto& immRenderImgView= immRenderTargetViews.emplace_back();
    createImageView(
            postProcFormat,
            vk::ImageAspectFlagBits::eColor,
            immRenderTarget.getImage(),
            immRenderImgView
    );
    setDebugObjectName(*immRenderImgView, "immediateRenderTargetImgView" + std::to_string(i));

    // for sampling imm render target
    auto& sampler = immRenderTargetSampler.emplace_back(device->createSamplerUnique(vk::SamplerCreateInfo(
            {},
            vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eNearest,
            vk::SamplerAddressMode::eClampToEdge, vk::SamplerAddressMode::eClampToEdge,
            vk::SamplerAddressMode::eClampToEdge,
            0.f, false, 0.0, false, vk::CompareOp::eNever, 0.0, 0.0, vk::BorderColor::eFloatTransparentBlack,
            true
    )));
    setDebugObjectName(*sampler, "immRtImgSampler" + std::to_string(i));


    // for pre-presentation (after post-processing)
    auto const& prePresentationTarget= prePresentationImages.emplace_back(createImage(
      backbufferExtent.width,
      backbufferExtent.height,
      vk::SampleCountFlagBits::e1,
      postProcFormat,
      vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eStorage,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    ));

    auto& prePresentationImageView= prePresentationImgViews.emplace_back();
    createImageView(
            postProcFormat,
            vk::ImageAspectFlagBits::eColor,
            prePresentationTarget.getImage(),
            prePresentationImageView
    );

    setDebugObjectName(vk::Image(prePresentationTarget.getImage()), "prePresentationTarget" + std::to_string(i));
    setDebugObjectName(*prePresentationImageView, "prePresentationImgView" + std::to_string(i));
  }
}

void AsyVkRender::createTransparencyBuffers(std::uint32_t pixels)
{
  std::uint32_t G=ceilquotient(pixels,groupSize);
  std::uint32_t Pixels=groupSize*G;
  globalSize=localSize*ceilquotient(G,localSize)*sizeof(std::uint32_t);

  countBufferSize=(Pixels+1)*sizeof(std::uint32_t);
  offsetBufferSize=(Pixels+2)*sizeof(std::uint32_t);
  opaqueBufferSize=pixels*sizeof(vec4);
  opaqueDepthBufferSize=pixels*sizeof(float);
  indexBufferSize=pixels*sizeof(std::uint32_t);

  VkMemoryPropertyFlags countBufferFlags=VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  VmaAllocationCreateFlags vmaFlags=0;

  if(fxaa) {
    countBufferFlags=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    vmaFlags=VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
  }

  countBf=createBufferUnique(
    vk::BufferUsageFlagBits::eStorageBuffer |
    vk::BufferUsageFlagBits::eTransferDst |
    vk::BufferUsageFlagBits::eTransferSrc,
    countBufferFlags,
    countBufferSize,
    vmaFlags,
    VMA_MEMORY_USAGE_AUTO,
    VARIABLE_NAME(countBf)
    );

  auto usageflags=vk::BufferUsageFlagBits::eStorageBuffer |
    vk::BufferUsageFlagBits::eTransferDst;

  globalSumBf=createBufferUnique(
      usageflags,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      globalSize,
      vmaFlags,
      VMA_MEMORY_USAGE_AUTO,
      VARIABLE_NAME(globalSumBf));

  offsetBf=createBufferUnique(
      usageflags,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      offsetBufferSize,
      vmaFlags,
      VMA_MEMORY_USAGE_AUTO,
      VARIABLE_NAME(offsetBf));

  opaqueBf=createBufferUnique(
      vk::BufferUsageFlagBits::eStorageBuffer,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      opaqueBufferSize,
      vmaFlags,
      VMA_MEMORY_USAGE_AUTO,
      VARIABLE_NAME(opaqueBf));

  opaqueDepthBf=createBufferUnique(
      vk::BufferUsageFlagBits::eStorageBuffer,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      opaqueDepthBufferSize,
      vmaFlags,
      VMA_MEMORY_USAGE_AUTO,
      VARIABLE_NAME(opaqueDepthBf));

  if (GPUcompress) {
    indexBf=createBufferUnique(
        usageflags,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        indexBufferSize,
        VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VARIABLE_NAME(indexBf));
  }

  zeroTransparencyBuffers();
  transparencyCapacityPixels=pixels;
}

void AsyVkRender::initIBL() {

  string imageDir=settings::locateFile(settings::getSetting<string>("imageDir"))+"/";
  string imagePath=imageDir+settings::getSetting<string>("image")+"/";

  auto const createReflectionSampler = [=](
    vma::cxx::UniqueImage& uniqueImg,
    vk::UniqueImageView& imageView,
    vk::UniqueSampler& sampler,
    std::vector<string> texturePaths
  ) {

    auto const imageType = texturePaths.size() > 1 ? vk::ImageType::e3D : vk::ImageType::e2D;
    auto const imageViewType = texturePaths.size() > 1 ? vk::ImageViewType::e3D : vk::ImageViewType::e2D;
    auto offset = 0;
    for (auto const& f: texturePaths) {

      camp::IEXRFile texture(f);

      auto && w = texture.size().first;
      auto && h = texture.size().second;

      if (uniqueImg.getImage() == VK_NULL_HANDLE) {

        uniqueImg = createImage(
                w, h,
                vk::SampleCountFlagBits::e1,
                vk::Format::eR32G32B32A32Sfloat,
                vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                imageType,
                texturePaths.size()
        );
        transitionImageLayout(vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, uniqueImg.getImage());
      }

      copyDataToImage(texture.getData(),
                      sizeof(vec4) * w * h,
                      uniqueImg.getImage(),
                      w, h,
                      {0, 0, offset++});
    }

    transitionImageLayout(vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, uniqueImg.getImage());
    createImageView(vk::Format::eR32G32B32A32Sfloat, vk::ImageAspectFlagBits::eColor, uniqueImg.getImage(), imageView, imageViewType);
    createImageSampler(sampler);
  };

  createReflectionSampler(
    irradianceImg,
    irradianceView,
    irradianceSampler,
    {imagePath+"diffuse.exr"}
  );

  createReflectionSampler(
    brdfImg,
    brdfView,
    brdfSampler,
    {imageDir+"refl.exr"}
  );

  std::vector<string> files;

  constexpr auto NTEXTURES=11;
  for(auto i = 0; i < NTEXTURES; ++i) {

    files.emplace_back(imagePath+"refl"+std::to_string(i).c_str()+".exr");
  }

  createReflectionSampler(
    reflectionImg,
    reflectionView,
    reflectionSampler,
    files
  );
}

void AsyVkRender::createCountRenderPass()
{
  std::array<vk::SubpassDescription2, 3> subpasses;

  subpasses[0] = vk::SubpassDescription2(
    vk::SubpassDescriptionFlags(),
    vk::PipelineBindPoint::eGraphics,
    0,
    0,
    nullptr,
    0,
    nullptr,
    nullptr,
    nullptr,
    0,
    nullptr,
    nullptr
  );
  subpasses[1] = vk::SubpassDescription2(
    vk::SubpassDescriptionFlags(),
    vk::PipelineBindPoint::eGraphics,
    0,
    0,
    nullptr,
    0,
    nullptr,
    nullptr,
    nullptr,
    0,
    nullptr,
    nullptr
  );
  subpasses[2] = vk::SubpassDescription2(
    vk::SubpassDescriptionFlags(),
    vk::PipelineBindPoint::eGraphics,
    0,
    0,
    nullptr,
    0,
    nullptr,
    nullptr,
    nullptr,
    0,
    nullptr,
    nullptr
  );

  std::array<vk::SubpassDependency2, 3> dependencies;

  dependencies[0] = vk::SubpassDependency2(
    VK_SUBPASS_EXTERNAL,
    0,
    vk::PipelineStageFlagBits::eColorAttachmentOutput,
    vk::PipelineStageFlagBits::eFragmentShader,
    vk::AccessFlagBits::eNone,
    vk::AccessFlagBits::eNone
  );
  dependencies[1] = vk::SubpassDependency2(
    0,
    1,
    vk::PipelineStageFlagBits::eFragmentShader,
    vk::PipelineStageFlagBits::eFragmentShader,
    vk::AccessFlagBits::eNone,
    vk::AccessFlagBits::eNone
  );
  dependencies[2] = vk::SubpassDependency2(
    1,
    2,
    vk::PipelineStageFlagBits::eBottomOfPipe,
    vk::PipelineStageFlagBits::eFragmentShader,
    vk::AccessFlagBits::eMemoryWrite,
    vk::AccessFlagBits::eMemoryRead
  );

  auto renderPassCI = vk::RenderPassCreateInfo2(
    vk::RenderPassCreateFlags(),
    0,
    nullptr,
    subpasses.size(),
    subpasses.data(),
    dependencies.size(),
    dependencies.data()
  );

  countRenderPass = device->createRenderPass2Unique(renderPassCI);

  if (!countRenderPass)
    runtimeError("failed to create the count render pass");
}

void AsyVkRender::createGraphicsRenderPass()
{
  auto colorAttachment = vk::AttachmentDescription2(
    vk::AttachmentDescriptionFlags(),
    postProcFormat,
    msaaSamples,
    vk::AttachmentLoadOp::eClear,
    vk::AttachmentStoreOp::eStore,
    vk::AttachmentLoadOp::eDontCare,
    vk::AttachmentStoreOp::eDontCare,
    vk::ImageLayout::eUndefined,
    vk::ImageLayout::eColorAttachmentOptimal
  );

  // If we are using fxaa, the output needs to be eGeneral
  // since we are passing that to fxaa compute shader, otherwise
  // we can go to presentSrc since we are passing it to the swap chain

  // Again, we should really be using scene graphs here. The
  // code will only get more complicated from now on...
  vk::ImageLayout colorAttachmentFinalLayout = fxaa ?
    vk::ImageLayout::eGeneral :
    (View ? vk::ImageLayout::ePresentSrcKHR :
     vk::ImageLayout::eColorAttachmentOptimal);


  auto colorResolveAttachment = vk::AttachmentDescription2(
    vk::AttachmentDescriptionFlags(),
    postProcFormat,
    vk::SampleCountFlagBits::e1,
    vk::AttachmentLoadOp::eDontCare,
    vk::AttachmentStoreOp::eStore,
    vk::AttachmentLoadOp::eDontCare,
    vk::AttachmentStoreOp::eDontCare,
    vk::ImageLayout::eUndefined,
          colorAttachmentFinalLayout
  );
  auto depthAttachment = vk::AttachmentDescription2(
    vk::AttachmentDescriptionFlags(),
    vk::Format::eD32Sfloat,
    msaaSamples,
    vk::AttachmentLoadOp::eClear,
    vk::AttachmentStoreOp::eDontCare,
    vk::AttachmentLoadOp::eDontCare,
    vk::AttachmentStoreOp::eDontCare,
    vk::ImageLayout::eUndefined,
    vk::ImageLayout::eDepthStencilAttachmentOptimal
  );

  auto colorAttachmentRef = vk::AttachmentReference2(0, vk::ImageLayout::eColorAttachmentOptimal);
  auto depthAttachmentRef = vk::AttachmentReference2(1, vk::ImageLayout::eDepthStencilAttachmentOptimal);
  auto colorResolveAttachmentRef= vk::AttachmentReference2(2, vk::ImageLayout::eColorAttachmentOptimal);

  std::vector subpasses{
          vk::SubpassDescription2(
                  {},
                  vk::PipelineBindPoint::eGraphics,
                  0,
                  0,
                  nullptr,
                  1,
                  &colorAttachmentRef,
                  &colorResolveAttachmentRef,
                  &depthAttachmentRef
          ),
          vk::SubpassDescription2({}, vk::PipelineBindPoint::eGraphics, 0, 0, nullptr, 0, nullptr, nullptr, nullptr),
          vk::SubpassDescription2({}, vk::PipelineBindPoint::eGraphics, 0, 0, nullptr, 1, &colorResolveAttachmentRef)
  };
  if (msaaSamples == vk::SampleCountFlagBits::e1)
  {
    colorAttachment.loadOp= vk::AttachmentLoadOp::eDontCare;
    colorResolveAttachment.loadOp = vk::AttachmentLoadOp::eClear;

    subpasses[0].pColorAttachments = &colorResolveAttachmentRef;
    subpasses[0].pResolveAttachments = nullptr;
  }

  std::vector const attachments
  {
    colorAttachment,
    depthAttachment,
    colorResolveAttachment
  };

  std::vector const dependencies{
          vk::SubpassDependency2(
                  VK_SUBPASS_EXTERNAL,
                  0,
                  vk::PipelineStageFlagBits::eColorAttachmentOutput,
                  vk::PipelineStageFlagBits::eColorAttachmentOutput,
                  vk::AccessFlagBits::eNone,
                  vk::AccessFlagBits::eNone
          ),
          vk::SubpassDependency2(
                  0,
                  2,
                  vk::PipelineStageFlagBits::eColorAttachmentOutput,
                  vk::PipelineStageFlagBits::eColorAttachmentOutput,
                  vk::AccessFlagBits::eNone,
                  vk::AccessFlagBits::eNone
          )
  };

  // only use the first subpass and first dependency
  auto const opaqueRenderPassCI=
          vk::RenderPassCreateInfo2({}, VEC_VIEW(attachments), 1, subpasses.data(), 1, dependencies.data());
  opaqueGraphicsRenderPass= device->createRenderPass2Unique(opaqueRenderPassCI);
  setDebugObjectName(*opaqueGraphicsRenderPass, "opaqueGraphicsRenderPass");

  if (!opaqueGraphicsRenderPass)
    runtimeError("failed to create the opaque render pass");

  auto renderPassCI = vk::RenderPassCreateInfo2(
    vk::RenderPassCreateFlags(),
    VEC_VIEW(attachments),
    VEC_VIEW(subpasses),
    VEC_VIEW(dependencies)
  );
  graphicsRenderPass = device->createRenderPass2Unique(renderPassCI);

  if (!graphicsRenderPass)
    runtimeError("failed to create the graphics render pass");
  setDebugObjectName(*graphicsRenderPass, "graphicsRenderPass");
}

void AsyVkRender::createGraphicsPipelineLayout()
{
  auto flagsPushConstant = vk::PushConstantRange(
    vk::ShaderStageFlagBits::eFragment,
    0,
    sizeof(PushConstants)
  );

  auto pipelineLayoutCI = vk::PipelineLayoutCreateInfo(
    vk::PipelineLayoutCreateFlags(),
    1,
    &*materialDescriptorSetLayout,
    1,
    &flagsPushConstant
  );

  graphicsPipelineLayout = device->createPipelineLayoutUnique(pipelineLayoutCI, nullptr);
}

void AsyVkRender::modifyShaderOptions(std::vector<std::string>& options, PipelineType type) {

  if (ibl) {
    options.emplace_back("USE_IBL");
  }
  if (orthographic) {
    options.emplace_back("ORTHOGRAPHIC");
  }

  if (fxaa)
  {
    options.emplace_back("ENABLE_FXAA");
  }

  if (srgb)
  {
    options.emplace_back("OUTPUT_AS_SRGB");
  }

  if (type == PIPELINE_OPAQUE) {
    options.emplace_back("OPAQUE");
    return;
  }

  // from now on, only things relevant to compute
  if (GPUcompress) {
    options.emplace_back("GPUCOMPRESS");
  }
  if (interlock) {
    options.emplace_back("HAVE_INTERLOCK");
  }

  options.emplace_back("LOCALSIZE " + std::to_string(localSize));
  options.emplace_back("BLOCKSIZE " + std::to_string(blockSize));
  options.emplace_back("ARRAYSIZE " + std::to_string(maxSize));
}

template<typename V>
void AsyVkRender::createGraphicsPipeline(PipelineType type, vk::UniquePipeline & graphicsPipeline, vk::PrimitiveTopology topology,
                                         vk::PolygonMode fillMode, std::vector<std::string> options,
                                         std::string const & vertexShader,
                                         std::string const & fragmentShader,
                                         int graphicsSubpass, bool enableDepthWrite,
                                         bool transparent, bool disableMultisample)
{
  std::string vertShaderName = SHADER_DIRECTORY + vertexShader + ".glsl";
  std::string fragShaderName = SHADER_DIRECTORY + fragmentShader + ".glsl";

  if (type == PIPELINE_COUNT) {
    fragShaderName = SHADER_DIRECTORY "count.glsl";
  }

  modifyShaderOptions(options, type);

  auto vertShaderModule = createShaderModule(EShLangVertex, vertShaderName, options);
  auto fragShaderModule = createShaderModule(EShLangFragment, fragShaderName, options);

  auto specializationInfo = vk::SpecializationInfo();

  auto vertShaderStageCI = vk::PipelineShaderStageCreateInfo(
    vk::PipelineShaderStageCreateFlags(),
    vk::ShaderStageFlagBits::eVertex,
    *vertShaderModule,
    "main",
    &specializationInfo
  );
  auto fragShaderStageCI = vk::PipelineShaderStageCreateInfo(
    vk::PipelineShaderStageCreateFlags(),
    vk::ShaderStageFlagBits::eFragment,
    *fragShaderModule,
    "main",
    &specializationInfo
  );

  vk::PipelineShaderStageCreateInfo stages[] = {vertShaderStageCI, fragShaderStageCI};

  auto bindingDescription = V::getBindingDescription();
  auto attributeDescriptions = V::getAttributeDescriptions();
  auto vertexInputCI = vk::PipelineVertexInputStateCreateInfo(
    vk::PipelineVertexInputStateCreateFlags(),
    1,
    &bindingDescription,
    VEC_VIEW(attributeDescriptions)
  );

  auto inputAssemblyCI = vk::PipelineInputAssemblyStateCreateInfo(
    vk::PipelineInputAssemblyStateCreateFlags(),
    topology,
    VK_FALSE
  );

  // Set origin at lower-left corner with y coordinate increasing up
  auto viewport = vk::Viewport(
    0.0f,
    static_cast<float>(backbufferExtent.height),
    static_cast<float>(backbufferExtent.width),
    -static_cast<float>(backbufferExtent.height),
    0.0f,
    1.0f
  );

  auto scissor = vk::Rect2D(
    vk::Offset2D(0, 0),
    backbufferExtent
  );

  auto viewportStateCI = vk::PipelineViewportStateCreateInfo(
    vk::PipelineViewportStateCreateFlags(),
    1,
    &viewport,
    1,
    &scissor
  );

  auto rasterizerCI = vk::PipelineRasterizationStateCreateInfo(
    vk::PipelineRasterizationStateCreateFlags(),
    VK_FALSE,
    VK_FALSE,
    fillMode,
    vk::CullModeFlagBits::eNone,
    vk::FrontFace::eCounterClockwise,
    VK_FALSE,
    0.0f,
    0.0f,
    0.0f,
    1.0f
  );

  auto multisamplingCI = vk::PipelineMultisampleStateCreateInfo(
    vk::PipelineMultisampleStateCreateFlags(),
    transparent || disableMultisample ? vk::SampleCountFlagBits::e1 : msaaSamples,
    VK_FALSE,
    0.0f,
    nullptr,
    VK_FALSE,
    VK_FALSE
  );

  auto colorBlendAttachment = vk::PipelineColorBlendAttachmentState(
    VK_FALSE,
    vk::BlendFactor::eZero,
    vk::BlendFactor::eZero,
    vk::BlendOp::eAdd,
    vk::BlendFactor::eZero,
    vk::BlendFactor::eZero,
    vk::BlendOp::eAdd,
    vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
  );

  auto colorBlendCI = vk::PipelineColorBlendStateCreateInfo(
    vk::PipelineColorBlendStateCreateFlags(),
    VK_FALSE,
    vk::LogicOp::eCopy,
    1,
    &colorBlendAttachment,
    {0.0f, 0.0f, 0.0f, 0.0f}
  );

  auto depthStencilCI = vk::PipelineDepthStencilStateCreateInfo();

  depthStencilCI.depthCompareOp = vk::CompareOp::eLess;
  depthStencilCI.depthBoundsTestEnable = VK_FALSE;
  depthStencilCI.minDepthBounds = 0.f;
  depthStencilCI.maxDepthBounds = 1.f;
  depthStencilCI.stencilTestEnable = VK_FALSE;

  vk::RenderPass renderPass;

  switch(type) {
    case PIPELINE_OPAQUE:
      renderPass=*opaqueGraphicsRenderPass;
      depthStencilCI.depthTestEnable=VK_TRUE;
      depthStencilCI.depthWriteEnable=enableDepthWrite;
      break;
    case PIPELINE_COUNT:
    case PIPELINE_COMPRESS:
      renderPass=*countRenderPass;
      depthStencilCI.depthTestEnable=VK_FALSE;
      depthStencilCI.depthWriteEnable=VK_FALSE;
      break;
    default:
      renderPass=*graphicsRenderPass;
      depthStencilCI.depthTestEnable=VK_TRUE;
      depthStencilCI.depthWriteEnable=enableDepthWrite;
      break;
  }

  auto pipelineCI = vk::GraphicsPipelineCreateInfo(
    vk::PipelineCreateFlags(),
    2,
    stages,
    &vertexInputCI,
    &inputAssemblyCI,
    nullptr,
    &viewportStateCI,
    &rasterizerCI,
    &multisamplingCI,
    &depthStencilCI,
    &colorBlendCI,
    nullptr,
    *graphicsPipelineLayout,
    renderPass,
    graphicsSubpass,
    nullptr
  );

  auto result = device->createGraphicsPipelineUnique(nullptr, pipelineCI, nullptr);
  if (result.result != vk::Result::eSuccess)
    runtimeError("failed to create pipeline");
  else
    graphicsPipeline = std::move(result.value);
}

void AsyVkRender::createGraphicsPipelines()
{
  auto const drawMode =
    (mode == DRAWMODE_WIREFRAME || mode == DRAWMODE_OUTLINE) ?
    vk::PolygonMode::eLine : vk::PolygonMode::eFill;

  for (auto u = 0u; u < PIPELINE_MAX; u++)
    createGraphicsPipeline<MaterialVertex>
                          (PipelineType(u), materialPipelines[u], vk::PrimitiveTopology::eTriangleList,
                          drawMode,
                          materialShaderOptions,
                          "vertex",
                          "fragment",
                          0);

  for (auto u = 0u; u < PIPELINE_MAX; u++)
    createGraphicsPipeline<ColorVertex>
                          (PipelineType(u), colorPipelines[u], vk::PrimitiveTopology::eTriangleList,
                          drawMode,
                          colorShaderOptions,
                          "vertex",
                          "fragment",
                          0);

  for (auto u = 0u; u < PIPELINE_MAX; u++)
    createGraphicsPipeline<ColorVertex>
                          (PipelineType(u), trianglePipelines[u], vk::PrimitiveTopology::eTriangleList,
                          drawMode,
                          triangleShaderOptions,
                          "vertex",
                          "fragment",
                          0);

  for (auto u = 0u; u < PIPELINE_MAX; u++)
    createGraphicsPipeline<MaterialVertex>
                          (PipelineType(u), linePipelines[u], vk::PrimitiveTopology::eLineList,
                          vk::PolygonMode::eLine,
                          materialShaderOptions,
                          "vertex",
                          "fragment",
                          0);

  for (auto u = 0u; u < PIPELINE_MAX; u++)
    createGraphicsPipeline<PointVertex>
                          (PipelineType(u), pointPipelines[u], vk::PrimitiveTopology::ePointList,
#ifdef __APPLE__
                          vk::PolygonMode::eFill,
#else
                          vk::PolygonMode::ePoint,
#endif
                          pointShaderOptions,
                          "vertex",
                          "fragment",
                          0);

  for (unsigned u = PIPELINE_TRANSPARENT; u < PIPELINE_MAX; u++)
    createGraphicsPipeline<ColorVertex>
                          (PipelineType(u), transparentPipelines[u], vk::PrimitiveTopology::eTriangleList,
                          drawMode,
                          transparentShaderOptions,
                          "vertex",
                          "fragment",
                          1,
                          false,
                          true);

  createGraphicsPipeline<ColorVertex>
                        (PIPELINE_COMPRESS, compressPipeline, vk::PrimitiveTopology::eTriangleList,
                        vk::PolygonMode::eFill,
                        {},
                        "screen",
                        "compress",
                        2,
                        false,
                        false,
                        true);

  createBlendPipeline();
}

void AsyVkRender::setupPostProcessingComputeParameters()
{
// TODO: We should share this constant with the shader code & C++ side")
  uint32_t constexpr localGroupSize=20;

  postProcessThreadGroupCount.width=ceilquotient(backbufferExtent.width, localGroupSize);
  postProcessThreadGroupCount.height=ceilquotient(backbufferExtent.height, localGroupSize);
}

void AsyVkRender::createBlendPipeline() {

  createGraphicsPipeline<ColorVertex>
                        (PIPELINE_DONTCARE, blendPipeline, vk::PrimitiveTopology::eTriangleList,
                        vk::PolygonMode::eFill,
                        {},
                        "screen",
                        "blend",
                        2,
                        false,
                        false,
                        true);
}

void AsyVkRender::createComputePipeline(
  vk::UniquePipelineLayout& layout,
  vk::UniquePipeline& pipeline,
  std::string const& shaderFile,
  std::vector<vk::DescriptorSetLayout> const& descSetLayout
)
{
  auto const filename = SHADER_DIRECTORY + shaderFile + ".glsl";

  std::vector<std::string> options;

  modifyShaderOptions(options, PIPELINE_DONTCARE);

  vk::UniqueShaderModule computeShaderModule = createShaderModule(EShLangCompute, filename, options);

  auto computeShaderStageInfo = vk::PipelineShaderStageCreateInfo(
    vk::PipelineShaderStageCreateFlags(),
    vk::ShaderStageFlagBits::eCompute,
    *computeShaderModule,
    "main"
  );

  auto miscConstant = vk::PushConstantRange(
    vk::ShaderStageFlagBits::eCompute,
    0,
    sizeof(ComputePushConstants)
  );

  auto pipelineLayoutCI = vk::PipelineLayoutCreateInfo(
    vk::PipelineLayoutCreateFlags(),
    VEC_VIEW(descSetLayout),
    0,
    nullptr
  );

  pipelineLayoutCI.pPushConstantRanges = &miscConstant;
  pipelineLayoutCI.pushConstantRangeCount = 1;

  layout = device->createPipelineLayoutUnique(pipelineLayoutCI, nullptr);

  auto computePipelineCI = vk::ComputePipelineCreateInfo();

  computePipelineCI.layout = *layout;
  computePipelineCI.stage = computeShaderStageInfo;

auto result = device->createComputePipelineUnique(VK_NULL_HANDLE, computePipelineCI);
  if (result.result != vk::Result::eSuccess)
    runtimeError("failed to create compute pipeline");
  else
    pipeline = std::move(result.value);
}

void AsyVkRender::createComputePipelines()
{
  std::vector const computeDescSetLayoutVec { *computeDescriptorSetLayout };
  createComputePipeline(sumPipelineLayout, sum1Pipeline, "sum1", computeDescSetLayoutVec);
  createComputePipeline(sumPipelineLayout, sum2Pipeline, "sum2", computeDescSetLayoutVec);
  createComputePipeline(sumPipelineLayout, sum3Pipeline, "sum3", computeDescSetLayoutVec);

  if (fxaa)
  {
    std::vector const postProcessDescSetLayoutVec{*postProcessDescSetLayout};
    // fxaa
    createComputePipeline(postProcessPipelineLayout, postProcessPipeline, "fxaa.cs", postProcessDescSetLayoutVec);
  }
}

void AsyVkRender::createAttachments()
{
  colorImg = createImage(backbufferExtent.width, backbufferExtent.height, msaaSamples, postProcFormat,
              vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment,
              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  createImageView(postProcFormat, vk::ImageAspectFlagBits::eColor, colorImg.getImage(), colorImageView);
  setDebugObjectName(vk::Image(colorImg.getImage()), "colorImg");
  setDebugObjectName(*colorImageView, "colorImageView");

  depthImg = createImage(backbufferExtent.width, backbufferExtent.height, msaaSamples, vk::Format::eD32Sfloat,
          vk::ImageUsageFlagBits::eDepthStencilAttachment,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
  );
  createImageView(vk::Format::eD32Sfloat, vk::ImageAspectFlagBits::eDepth, depthImg.getImage(), depthImageView);
  setDebugObjectName(vk::Image(depthImg.getImage()), "depthImg");
  setDebugObjectName(*depthImageView, "depthImageView");

  depthResolveImg = createImage(backbufferExtent.width, backbufferExtent.height, vk::SampleCountFlagBits::e1, vk::Format::eD32Sfloat,
          vk::ImageUsageFlagBits::eDepthStencilAttachment,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
  );
  createImageView(vk::Format::eD32Sfloat, vk::ImageAspectFlagBits::eDepth, depthResolveImg.getImage(), depthResolveImageView);
  setDebugObjectName(vk::Image(depthResolveImg.getImage()), "depthResolve");
  setDebugObjectName(*depthResolveImageView, "depthResolveImageView");
}

void AsyVkRender::updateUniformBuffer(uint32_t currentFrame)
{
  if (!newUniformBuffer && !queueExport)
    return;

  UniformBufferObject ubo{ };

  ubo.projViewMat = projViewMat;
  ubo.viewMat = viewMat;
  ubo.normMat = normMat;

  memcpy(frameObjects[currentFrame].uboMappedMemory->getCopyPtr(), &ubo, sizeof(ubo));

  newUniformBuffer = false;
}

void AsyVkRender::updateBuffers()
{
  // Don't update the material buffer if the materials aren't yet added
  bool materialsReady = !materials.empty();

  if (shouldUpdateBuffers && materialsReady) {
    std::vector<Light> lights;

    for (auto i = 0u; i < nlights; i++)
      lights.emplace_back(
        Light {
          {Lights[i].getx(), Lights[i].gety(), Lights[i].getz(), 0.f},
          {static_cast<float>(LightsDiffuse[4 * i]),
          static_cast<float>(LightsDiffuse[4 * i + 1]),
          static_cast<float>(LightsDiffuse[4 * i + 2]), 0.f}
        }
      );


    if (materials.size() > nmaterials) {
      nmaterials=materials.size();
    }

    createMaterialAndLightBuffers();
    writeMaterialAndLightDescriptors();

    if(lights.size() > 0)
      copyToBuffer(lightBf.getBuffer(), lights.data(), lights.size() * sizeof(Light));
    if(materials.size() > 0)
      copyToBuffer(materialBf.getBuffer(), materials.data(), materials.size() * sizeof(camp::Material));

    shouldUpdateBuffers=false;
  }
}

PushConstants AsyVkRender::buildPushConstants()
{
  auto pushConstants = PushConstants {};

  pushConstants.constants[0] = mode!= DRAWMODE_NORMAL ? 0 : nlights;
  pushConstants.constants[1] = backbufferExtent.width;
  pushConstants.constants[2] = backbufferExtent.height;

  for (int i = 0; i < 4; i++)
    pushConstants.background[i]=Background[i];

  return pushConstants;
}

vk::CommandBuffer & AsyVkRender::getFrameCommandBuffer()
{
  return *frameObjects[currentFrame].commandBuffer;
}

vk::CommandBuffer & AsyVkRender::getFrameComputeCommandBuffer()
{
  return *frameObjects[currentFrame].computeCommandBuffer;
}

vk::UniquePipeline & AsyVkRender::getPipelineType(std::array<vk::UniquePipeline, PIPELINE_MAX> & pipelines)
{
  return pipelines[Opaque ? PIPELINE_OPAQUE : PIPELINE_TRANSPARENT];
}

void AsyVkRender::beginFrameCommands(vk::CommandBuffer cmd)
{
  currentCommandBuffer = cmd;
  currentCommandBuffer.begin(vk::CommandBufferBeginInfo());
}

void AsyVkRender::beginCountFrameRender(int imageIndex)
{
  std::vector<vk::ClearValue> clearColors;


  auto renderPassInfo = vk::RenderPassBeginInfo(
    *countRenderPass,
    *depthFramebuffers[imageIndex],
    vk::Rect2D(vk::Offset2D(0, 0), backbufferExtent),
    clearColors.size(),
    clearColors.data()
  );

  currentCommandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
}

void AsyVkRender::beginGraphicsFrameRender(int imageIndex)
{
  std::array<vk::ClearValue, 3> clearColors;

  clearColors[0]= vk::ClearValue(Background);
  clearColors[1].depthStencil.depth = 1.f;
  clearColors[1].depthStencil.stencil = 0;
  clearColors[2] = vk::ClearValue(Background);

  auto renderPassInfo = vk::RenderPassBeginInfo(
    Opaque ? *opaqueGraphicsRenderPass : *graphicsRenderPass,
    Opaque ? *opaqueGraphicsFramebuffers[imageIndex] : *graphicsFramebuffers[imageIndex],
    vk::Rect2D(vk::Offset2D(0, 0), backbufferExtent),
    clearColors.size(),
    &clearColors[0]
  );

  currentCommandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
}

void AsyVkRender::resetFrameCopyData()
{
  materialData.copiedThisFrame=false;
  colorData.copiedThisFrame=false;
  triangleData.copiedThisFrame=false;
  transparentData.copiedThisFrame=false;
  lineData.copiedThisFrame=false;
  pointData.copiedThisFrame=false;
}

void AsyVkRender::drawBuffer(DeviceBuffer & vertexBuffer,
                             DeviceBuffer & indexBuffer,
                             VertexBuffer * data,
                             vk::UniquePipeline & pipeline,
                             bool incrementRenderCount) {
  if (data->indices.empty())
    return;

  auto const badBuffer = static_cast<void*>(vertexBuffer._buffer.getBuffer()) == nullptr;
  auto const rendered = data->renderCount >= maxFramesInFlight;
  auto const copy = (remesh || !rendered || badBuffer) && !copied && !data->copiedThisFrame;

  if (copy) {

    if (!data->materialVertices.empty())
    {
      setDeviceBufferData(vertexBuffer, data->materialVertices.data(), data->materialVertices.size() * sizeof(camp::MaterialVertex));
    }
    else if (!data->colorVertices.empty())
    {
      setDeviceBufferData(vertexBuffer, data->colorVertices.data(), data->colorVertices.size() * sizeof(camp::ColorVertex));
    }
    else if(!data->pointVertices.empty())
    {
      setDeviceBufferData(vertexBuffer, data->pointVertices.data(), data->pointVertices.size() * sizeof(camp::PointVertex));
    }
    else
      return;

    setDeviceBufferData(indexBuffer, data->indices.data(), data->indices.size() * sizeof(data->indices[0]), data->indices.size());
    data->copiedThisFrame=true;
  }

  std::vector<vk::Buffer> vertexBuffers = {vertexBuffer._buffer.getBuffer()};
  std::vector<vk::DeviceSize> vertexOffsets = {0};
  auto const pushConstants = buildPushConstants();

  currentCommandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline);
  currentCommandBuffer.bindVertexBuffers(0, vertexBuffers, vertexOffsets);
  currentCommandBuffer.bindIndexBuffer(indexBuffer._buffer.getBuffer(), 0, vk::IndexType::eUint32);
  currentCommandBuffer.pushConstants(*graphicsPipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(PushConstants), &pushConstants);
  currentCommandBuffer.drawIndexed(indexBuffer.nobjects, 1, 0, 0, 0);

  if(incrementRenderCount)
    data->renderCount++;
}

void AsyVkRender::endFrameRender()
{
  currentCommandBuffer.endRenderPass();
}

void AsyVkRender::endFrameCommands()
{
  currentCommandBuffer.end();
}

void AsyVkRender::endFrame(int imageIndex)
{
  endFrameRender();
  endFrameCommands();
}

void AsyVkRender::clearData()
{
  pointData.clear();
  lineData.clear();
  materialData.clear();
  colorData.clear();
  triangleData.clear();
  transparentData.clear();
}

void AsyVkRender::drawPoints(FrameObject & object)
{
  drawBuffer(object.pointVertexBuffer,
             object.pointIndexBuffer,
             &pointData,
             getPipelineType(pointPipelines));
}

void AsyVkRender::drawLines(FrameObject & object)
{
  drawBuffer(object.lineVertexBuffer,
             object.lineIndexBuffer,
             &lineData,
             getPipelineType(linePipelines));
}

void AsyVkRender::drawMaterials(FrameObject & object)
{
  drawBuffer(object.materialVertexBuffer,
             object.materialIndexBuffer,
             &materialData,
             getPipelineType(materialPipelines));
}

void AsyVkRender::drawColors(FrameObject & object)
{
  drawBuffer(object.colorVertexBuffer,
             object.colorIndexBuffer,
             &colorData,
             getPipelineType(colorPipelines));
}

void AsyVkRender::drawTriangles(FrameObject & object)
{
  drawBuffer(object.triangleVertexBuffer,
             object.triangleIndexBuffer,
             &triangleData,
             getPipelineType(trianglePipelines));
}

void AsyVkRender::drawTransparent(FrameObject & object)
{
  drawBuffer(object.transparentVertexBuffer,
             object.transparentIndexBuffer,
             &transparentData,
             getPipelineType(transparentPipelines));
}

void AsyVkRender::partialSums(FrameObject & object, bool timing)
{
  auto const writeBarrier=vk::MemoryBarrier(
    vk::AccessFlagBits::eShaderWrite,
    vk::AccessFlagBits::eShaderRead
  );

  vk::CommandBuffer const cmd=timing ? *object.partialSumsCommandBuffer :
    currentCommandBuffer;

  auto const blockSize=ceilquotient(g,localSize);
  auto const final=elements-1;
  ComputePushConstants pc{blockSize, final};

  cmd.pushConstants(*sumPipelineLayout,vk::ShaderStageFlagBits::eCompute,0,
                    sizeof(ComputePushConstants),&pc);

  cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute,*sumPipelineLayout,
                         0,1,&*computeDescriptorSet,0,nullptr);

  // run sum1
  // Only wait for fragment shaders if we are not timing
  if(!timing)
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eFragmentShader,
                        vk::PipelineStageFlagBits::eComputeShader,
                        { },
                        1,
                        &writeBarrier,
                        0,
                        nullptr,
                        0,
                        nullptr);

  cmd.bindPipeline(vk::PipelineBindPoint::eCompute,*sum1Pipeline);
  cmd.dispatch(g,1,1);

  // run sum2
  cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                      vk::PipelineStageFlagBits::eComputeShader,
                      { },
                      1,
                      &writeBarrier,
                      0,
                      nullptr,
                      0,
                      nullptr);
  cmd.bindPipeline(vk::PipelineBindPoint::eCompute,*sum2Pipeline);
  cmd.dispatch(1,1,1);

  // run sum3
  cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                      vk::PipelineStageFlagBits::eComputeShader,
                      { },
                      1,
                      &writeBarrier,
                      0,
                      nullptr,
                      0,
                      nullptr);
  cmd.bindPipeline(vk::PipelineBindPoint::eCompute,*sum3Pipeline);
  cmd.dispatch(g,1,1);

  if(timing)
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                        vk::PipelineStageFlagBits::eComputeShader,
                        { },
                        1,
                        &writeBarrier,
                        0,
                        nullptr,
                        0,
                        nullptr);
  else
    cmd.setEvent(*object.sumFinishedEvent,
                 vk::PipelineStageFlagBits::eComputeShader);
}

void AsyVkRender::resizeBlendShader(std::uint32_t maxDepth) {

  maxSize=maxDepth;

  recreateBlendPipeline=true;
}

void AsyVkRender::resizeFragmentBuffer(FrameObject & object) {
  waitForEvent(*object.sumFinishedEvent);

  static const auto feedbackMappedPtr=make_unique<vma::cxx::MemoryMapperLock>(feedbackBf);

  std::uint32_t maxDepth=feedbackMappedPtr->getCopyPtr()[0];

  fragments=feedbackMappedPtr->getCopyPtr()[1];

  if(resetDepth) {
    maxSize=maxDepth=1;
    resetDepth=false;
  }

  if (maxDepth>maxSize) {
    resizeBlendShader(maxDepth);
  }

  if (fragments>maxFragments) {
    maxFragments=11*fragments/10;
    device->waitIdle();
    updateSceneDependentBuffers();
  }
}

void AsyVkRender::compressCount(FrameObject & object)
{
  auto push = buildPushConstants();
  currentCommandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *compressPipeline);
  currentCommandBuffer.pushConstants(*graphicsPipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(PushConstants), &push);
  currentCommandBuffer.draw(3, 1, 0, 0);
}

void AsyVkRender::refreshBuffers(FrameObject & object, int imageIndex) {
  std::vector<vk::CommandBuffer> commandsToSubmit {};

  beginFrameCommands(*object.countCommandBuffer);

  currentCommandBuffer.fillBuffer(countBf.getBuffer(), 0, countBufferSize, 0);

  beginCountFrameRender(imageIndex);
  currentCommandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *graphicsPipelineLayout, 0, 1, &*object.descriptorSet, 0, nullptr);

  if (!interlock) {
    drawBuffer(object.pointVertexBuffer,
               object.pointIndexBuffer,
               &pointData,
               pointPipelines[PIPELINE_COUNT],
               false);
    drawBuffer(object.lineVertexBuffer,
               object.lineIndexBuffer,
               &lineData,
               linePipelines[PIPELINE_COUNT],
               false);
    drawBuffer(object.materialVertexBuffer,
               object.materialIndexBuffer,
               &materialData,
               materialPipelines[PIPELINE_COUNT],
               false);
    drawBuffer(object.colorVertexBuffer,
               object.colorIndexBuffer,
               &colorData,
               colorPipelines[PIPELINE_COUNT],
               false);
    drawBuffer(object.triangleVertexBuffer,
               object.triangleIndexBuffer,
               &triangleData,
               trianglePipelines[PIPELINE_COUNT],
               false);
  }

  currentCommandBuffer.nextSubpass(vk::SubpassContents::eInline);

  // draw transparent
  drawBuffer(object.transparentVertexBuffer,
             object.transparentIndexBuffer,
             &transparentData,
             transparentPipelines[PIPELINE_COUNT],
             false);

  currentCommandBuffer.nextSubpass(vk::SubpassContents::eInline);

  if (GPUcompress) {
    static auto elemBfMappedMem=make_unique<vma::cxx::MemoryMapperLock>(elementBf);
    static std::uint32_t* p = nullptr;

    if (p == nullptr) {
      p=elemBfMappedMem->getCopyPtr();
      *p=1;
    }

    compressCount(object);
    endFrameRender();
    currentCommandBuffer.setEvent(*object.compressionFinishedEvent, vk::PipelineStageFlagBits::eFragmentShader);
    endFrameCommands();

    // Create a fence for synchronization
    auto compressFence = device->createFenceUnique(vk::FenceCreateInfo());

    // Submit the command buffer with the fence
    auto submitInfo = vk::SubmitInfo();
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &currentCommandBuffer;

    vkutils::checkVkResult(renderQueue.submit(1, &submitInfo, *compressFence));

    // Wait for the fence with a reasonable timeout
    vkutils::checkVkResult(device->waitForFences(
      1, &*compressFence, VK_TRUE, timeout
    ));

    elements=p[0];
    p[0]=1;
  } else {
    endFrameRender();
    endFrameCommands();
    elements=pixels;
    commandsToSubmit.emplace_back(currentCommandBuffer);
  }

  if (elements==0)
    return;

  beginFrameCommands(*object.computeCommandBuffer);
  g=ceilquotient(elements,groupSize);
  elements=groupSize*g;

  const unsigned int NSUMS=10000;

  if(settings::verbose >= timePartialSumVerbosity) {
    cerr << "Timing partial sums:" << endl;
    device->resetEvent(*object.startTimedSumsEvent);
    device->resetEvent(*object.timedSumsFinishedEvent);
    // Start recording commands into partialSumsCommandBuffer
    object.partialSumsCommandBuffer->begin(vk::CommandBufferBeginInfo());

    // Wait to execute the compute shaders until we trigger them from CPU
    object.partialSumsCommandBuffer->waitEvents(
      1,
      &*object.startTimedSumsEvent,
      vk::PipelineStageFlagBits::eHost,
      vk::PipelineStageFlagBits::eComputeShader,
      0,
      nullptr,
      0,
      nullptr,
      0,
      nullptr
      );

    // Record all partial sums calcs into partialSumsCommandBuffer
    for(unsigned int i=0; i < NSUMS; ++i)
      partialSums(object,true);

    // Signal to the CPU once the compute shaders have executed
    object.partialSumsCommandBuffer->setEvent(*object.timedSumsFinishedEvent, vk::PipelineStageFlagBits::eComputeShader);
    object.partialSumsCommandBuffer->end();
  }

  partialSums(object);
  endFrameCommands();
  commandsToSubmit.emplace_back(currentCommandBuffer);

  // This submission is for the transparency pre-computation (count + partial sums).
  // It MUST be synchronized with a fence because the CPU needs to read back the results
  // in resizeFragmentBuffer before the main graphics pass can be recorded.
  auto info = vk::SubmitInfo();
  info.commandBufferCount = commandsToSubmit.size();
  info.pCommandBuffers = commandsToSubmit.data();
  vkutils::checkVkResult(renderQueue.submit(1, &info, *object.inComputeFence));

  if(settings::verbose >= timePartialSumVerbosity) {
    // Wait until the render queue isn't being used, so we only time
    // our partial sums calculation
    renderQueue.waitIdle();

    auto partialSumsInfo = vk::SubmitInfo();

    partialSumsInfo.commandBufferCount = 1;
    partialSumsInfo.pCommandBuffers = &*object.partialSumsCommandBuffer;

    // Signal GPU to start partial sums
    device->setEvent(*object.startTimedSumsEvent);

    // Send all the partial sum commands to the GPU.
    vkutils::checkVkResult(renderQueue.submit(1, &partialSumsInfo, nullptr));

    // Start recording the time
    utils::stopWatch Timer;

    // Wait until the GPU tells us the sums are finished
    waitForEvent(*object.timedSumsFinishedEvent);

    // End recording
    double T=Timer.seconds()/NSUMS;
    cout << "elements=" << elements << endl;
    cout << "T (ms)=" << T*1e3 << endl;
    cout << "Megapixels/second=" << elements/T/1e6 << endl;
  }
}

void AsyVkRender::blendFrame(int imageIndex)
{
  auto push = buildPushConstants();
  currentCommandBuffer.bindPipeline(
    vk::PipelineBindPoint::eGraphics,
    *blendPipeline
  );
  currentCommandBuffer.pushConstants(*graphicsPipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(PushConstants), &push);
  currentCommandBuffer.draw(3, 1, 0, 0);
}

void AsyVkRender::preDrawBuffers(FrameObject & object, int imageIndex)
{
  copied=false;

  if(!Opaque) {
    vkutils::checkVkResult(device->waitForFences(
      1, &*object.inComputeFence, VK_TRUE, timeout
    ));

    pixels=backbufferExtent.width*backbufferExtent.height;
    if(pixels > transparencyCapacityPixels) {
      device->waitIdle();
      createTransparencyBuffers(pixels);
      writeDescriptorSets(true);
    }

    vkutils::checkVkResult(device->resetFences(
      1, &*object.inComputeFence
    ));
    device->resetEvent(*object.sumFinishedEvent);
    device->resetEvent(*object.compressionFinishedEvent);

    object.countCommandBuffer->reset();
    object.computeCommandBuffer->reset();

    refreshBuffers(object, imageIndex);
    resizeFragmentBuffer(object);
  }
}

void AsyVkRender::drawBuffers(FrameObject & object, int imageIndex)
{
  beginGraphicsFrameRender(imageIndex);
  currentCommandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *graphicsPipelineLayout, 0, 1, &*object.descriptorSet, 0, nullptr);
  drawPoints(object);
  drawLines(object);
  drawMaterials(object);
  drawColors(object);
  drawTriangles(object);

  if(!Opaque) {
    currentCommandBuffer.nextSubpass(vk::SubpassContents::eInline);
    drawTransparent(object);
    currentCommandBuffer.nextSubpass(vk::SubpassContents::eInline);
    blendFrame(imageIndex);
  }

  endFrameRender();
}

void AsyVkRender::postProcessImage(vk::CommandBuffer& cmdBuffer, uint32_t const& frameIndex)
{
  cmdBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *postProcessPipeline);

  std::vector const computeDescSet{*postProcessDescSet[frameIndex]};
  cmdBuffer.bindDescriptorSets(
          vk::PipelineBindPoint::eCompute,
          *postProcessPipelineLayout,
          0,
          VEC_VIEW(computeDescSet),
          EMPTY_VIEW
  );
  cmdBuffer.dispatch(postProcessThreadGroupCount.width, postProcessThreadGroupCount.height, 1);
}

void AsyVkRender::copyToSwapchainImg(vk::CommandBuffer& cmdBuffer, uint32_t const& frameIndex)
{
  // Formats differ (pre-presentation RGBA8 -> swapchain BGRA8), use blit
  vk::ImageBlit blit{};
  blit.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
  blit.srcOffsets[0] = vk::Offset3D(0, 0, 0);
  blit.srcOffsets[1] =  vk::Offset3D(
    static_cast<int32_t>(backbufferExtent.width), static_cast<int32_t>(backbufferExtent.height), 1);
  blit.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
  blit.dstOffsets[0] = vk::Offset3D(0, 0, 0);
  blit.dstOffsets[1] =  vk::Offset3D(
    static_cast<int32_t>(backbufferExtent.width), static_cast<int32_t>(backbufferExtent.height), 1);

  cmdBuffer.blitImage(
    prePresentationImages[frameIndex].getImage(),
    vk::ImageLayout::eTransferSrcOptimal,
    backbufferImages[frameIndex],
    vk::ImageLayout::eTransferDstOptimal,
    1, &blit, vk::Filter::eNearest
    );
}

void AsyVkRender::drawFrame()
{
  auto& frameObject = frameObjects[currentFrame];

  if (timelineSemaphoreSupported) {
    // Wait only if we are about to reuse a frame that is still in use by the GPU.
    // We check if the timeline value for this specific frame has been reached.
    if (frameObject.timelineValue > 0) {
      waitForTimelineSemaphore(*renderTimelineSemaphore, frameObject.timelineValue, timeout);
    }
  } else {
    // Fallback to the original fence-based synchronization
    vkutils::checkVkResult(device->waitForFences(1, &*frameObject.inFlightFence, VK_TRUE, timeout));
  }

  if (recreatePipeline)
  {
    device->waitIdle();
    recreatePipeline = false;
    createGraphicsPipelines();
  }

  uint32_t imageIndex = 0;
  if (View) {
    auto const result = device->acquireNextImageKHR(*swapChain, timeout, *frameObject.imageAvailableSemaphore, nullptr, &imageIndex);
    if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR || framebufferResized) {
      framebufferResized = false;
      recreateSwapChain();
      return;
    }
    else if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR)
      runtimeError("failed to acquire next swapchain image");
  }

  if (!timelineSemaphoreSupported) {
      vkutils::checkVkResult(device->resetFences(1, &*frameObject.inFlightFence));
  }

  frameObject.commandBuffer->reset(vk::CommandBufferResetFlagBits());

  updateUniformBuffer(currentFrame);
  updateBuffers();
  resetFrameCopyData();
  preDrawBuffers(frameObject, imageIndex);

  beginFrameCommands(getFrameCommandBuffer());
  drawBuffers(frameObject, imageIndex);
  if (fxaa) {
    auto& cmdBuffer = *frameObject.commandBuffer;

    // Transition immediate render target to general layout for compute shader access
    transitionImageLayout(
      cmdBuffer,
      immediateRenderTargetImgs[imageIndex].getImage(),
      vk::AccessFlagBits::eColorAttachmentWrite,
      vk::AccessFlagBits::eShaderRead,
      vk::ImageLayout::eColorAttachmentOptimal,
      vk::ImageLayout::eGeneral,
      vk::PipelineStageFlagBits::eColorAttachmentOutput,
      vk::PipelineStageFlagBits::eComputeShader,
      vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)
    );

    // Run FXAA compute shader
    postProcessImage(cmdBuffer, imageIndex);

    // Prepare for presentation
    transitionImageLayout(
      cmdBuffer,
      prePresentationImages[imageIndex].getImage(),
      vk::AccessFlagBits::eShaderWrite,
      vk::AccessFlagBits::eTransferRead,
      vk::ImageLayout::eGeneral,
      vk::ImageLayout::eTransferSrcOptimal,
      vk::PipelineStageFlagBits::eComputeShader,
      vk::PipelineStageFlagBits::eTransfer,
      vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)
    );
    copyToSwapchainImg(cmdBuffer, imageIndex);
  }
  endFrameCommands();

  std::vector<vk::Semaphore> waitSems;
  std::vector<uint64_t> waitSemaphoreValues;
  std::vector<vk::PipelineStageFlags> waitStages;
  if (View) {
      waitSems.push_back(*frameObject.imageAvailableSemaphore);
      waitStages.push_back(vk::PipelineStageFlagBits::eColorAttachmentOutput);
  }

  std::vector<vk::Semaphore> signalSems;
  if (View) {
      signalSems.push_back(*frameObject.renderFinishedSemaphore);
  }

  vk::SubmitInfo submitInfo;
  submitInfo.pWaitSemaphores = waitSems.data();
  submitInfo.waitSemaphoreCount = waitSems.size();
  submitInfo.pWaitDstStageMask = waitStages.data();
  submitInfo.pCommandBuffers = &*frameObject.commandBuffer;
  submitInfo.commandBufferCount = 1;

  vk::TimelineSemaphoreSubmitInfo timelineInfo;
  std::vector<uint64_t> signalValues;

  if (timelineSemaphoreSupported && !waitSems.empty()) {
    // Add wait values for binary semaphores (0)
    waitSemaphoreValues.resize(waitSems.size(), 0);
    timelineInfo.waitSemaphoreValueCount = waitSemaphoreValues.size();
    timelineInfo.pWaitSemaphoreValues = waitSemaphoreValues.data();
  }

  if (timelineSemaphoreSupported) {
      currentTimelineValue++;
      frameObject.timelineValue = currentTimelineValue;

      signalSems.push_back(*renderTimelineSemaphore);

      // The value for the binary semaphore is ignored, but the count must match.
      if (View) {
          signalValues.push_back(0);
      }
      signalValues.push_back(frameObject.timelineValue);

      timelineInfo.signalSemaphoreValueCount = signalValues.size();
      timelineInfo.pSignalSemaphoreValues = signalValues.data();
      submitInfo.pNext = &timelineInfo;

      submitInfo.pSignalSemaphores = signalSems.data();
      submitInfo.signalSemaphoreCount = signalSems.size();

      vkutils::checkVkResult(renderQueue.submit(1, &submitInfo, nullptr));
  } else {
      submitInfo.pSignalSemaphores = signalSems.data();
      submitInfo.signalSemaphoreCount = signalSems.size();
      vkutils::checkVkResult(renderQueue.submit(1, &submitInfo, *frameObject.inFlightFence));
  }

  if (View) {
    // The presentation engine only needs to wait on the binary semaphore.
    std::vector<vk::Semaphore> presentWaitSemaphores;
    presentWaitSemaphores.push_back(*frameObject.renderFinishedSemaphore);

    try {
      auto presentInfo = vk::PresentInfoKHR(VEC_VIEW(presentWaitSemaphores), 1, &*swapChain, &imageIndex);
      auto const result = presentQueue.presentKHR(presentInfo);

      if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR || framebufferResized) {
        framebufferResized = false;
        recreateSwapChain();
      } else if (result != vk::Result::eSuccess) {
        runtimeError("failed to present swapchain image");
      }
    } catch (std::exception const & e) {
      auto what=std::string(e.what());
      if (what.find("ErrorOutOfDateKHR") != std::string::npos) {
        framebufferResized = false;
        recreateSwapChain();
      } else {
        runtimeError(what);
      }
    }
  }

  if(queueExport) {
    // Wait for the just-submitted frame to finish before exporting
    if (timelineSemaphoreSupported) {
      waitForTimelineSemaphore(*renderTimelineSemaphore, frameObject.timelineValue);
    } else {
      vkutils::checkVkResult(device->waitForFences(1, &*frameObject.inFlightFence, VK_TRUE, timeout));
    }
    Export(imageIndex);
    queueExport=false;
  }

  if (recreateBlendPipeline) {
    if (timelineSemaphoreSupported) {
      waitForTimelineSemaphore(*renderTimelineSemaphore, frameObject.timelineValue);
    } else {
      vkutils::checkVkResult(device->waitForFences(1, &*frameObject.inFlightFence, VK_TRUE, timeout));
    }
    createBlendPipeline();
    recreateBlendPipeline=false;
  }

  currentFrame = (currentFrame + 1) % maxFramesInFlight;
}

void AsyVkRender::nextFrame()
{
#ifdef HAVE_PTHREAD
  endwait(readySignal,readyLock);
#endif
  double delay=settings::getSetting<double>("framerate");
  if(delay != 0.0) delay=1.0/delay;
  double seconds=frameTimer.seconds(true);
  delay -= seconds;
  if(delay > 0) {
    std::this_thread::sleep_for(std::chrono::duration<double>(delay));
  }
  if(Step) Animate=false;
}

void AsyVkRender::clearBuffers()
{
  // Get the most recent frame that was started and wait for it
  // to finish before clearing buffers
  int previousFrameIndex = currentFrame - 1;

  if (previousFrameIndex < 0) {
    previousFrameIndex = maxFramesInFlight - 1;
  }

  (void) device->waitForFences(1, &*frameObjects[previousFrameIndex].inFlightFence, VK_TRUE, timeout);

  for (int i = 0; i < maxFramesInFlight; i++) {
    frameObjects[i].reset();
  }
}

void AsyVkRender::render()
{

#ifdef HAVE_PTHREAD
  static bool first=true;
  if(vkthread && first) {
    wait(initSignal,initLock);
    endwait(initSignal,initLock);
    first=false;
  }

  if(format3dWait)
    wait(initSignal,initLock);
#endif

  if(redraw) {
    clearData();

    if(remesh)
      clearCenters();

    triple m(xmin,ymin,Zmin);
    triple M(xmax,ymax,Zmax);
    double perspective = orthographic ? 0.0 : 1.0 / Zmax;

    double size2=hypot(width,height);

    pic->render(size2,m,M,perspective,remesh);
    redraw=false;

    if(mode != DRAWMODE_OUTLINE)
      remesh=false;

    Opaque=transparentData.indices.empty();
  }
}

void AsyVkRender::display()
{
  render();

  if(View && !hideWindow && !glfwGetWindowAttrib(window,GLFW_VISIBLE))
    glfwShowWindow(window);

  drawFrame();

  bool fps=settings::verbose > 2;
  if(fps) {
    if(framecount < 20) // Measure steady-state framerate
      fpsTimer.reset();
    else {
      double s=fpsTimer.seconds(true);
      if(s > 0.0) {
        double rate=1.0/s;
        fpsStats.add(rate);
        if(framecount % 20 == 0)
          cout << "FPS=" << rate << "\t" << fpsStats.mean()
               << " +/- " << fpsStats.stdev() << endl;
      }
    }
    ++framecount;
  }

#ifdef HAVE_PTHREAD
  if(vkthread && Animate) {
    queueExport=false;
    nextFrame();
  }
#endif
  if(!vkthread) {
#if defined(_WIN32)
// TODO: Check if we need a threadless-based vk renderer
#else
    if(Oldpid != 0 && waitpid(Oldpid,NULL,WNOHANG) != Oldpid) {
      kill(Oldpid,SIGHUP);
      Oldpid=0;
    }
#endif
  }
}

void AsyVkRender::processMessages(VulkanRendererMessage const& msg)
{
  switch (msg)
  {
    case exportRender: {
      if (readyForExport)
      {
        readyForExport=false;
        exportHandler(0);
      }
    }
      break;
    case updateRenderer: {
      updateHandler(0);
    }
      break;
    default:
      break;
  }
}

void AsyVkRender::mainLoop()
{
  if(View) {
    while(!glfwWindowShouldClose(window)) {
      if(redraw || redisplay || queueExport)
        {
        redisplay=false;
        waitEvent=true;
        if(resize) {
          fitscreen(!interact::interactive);
          resize=false;
        }
        display();
      }

      auto const message=messageQueue.dequeue();
      if(message.has_value())
        processMessages(*message);

      if(currentIdleFunc != nullptr) {
        currentIdleFunc();
        glfwPollEvents();
      } else {
        if(waitEvent)
          glfwWaitEvents();
        else
          glfwPollEvents();
      }
    }
  } else {
    update();
    display();
    if(vkthread) {
      if(havewindow) {
#ifdef HAVE_PTHREAD
        if(pthread_equal(pthread_self(),this->mainthread))
          exportHandler();
        else
          messageQueue.enqueue(exportRender);
#endif
      } else {
        initialized=true;
        readyForExport=true;
        exportHandler();
      }
    } else {
      exportHandler();
      quit();
    }
  }
}

void AsyVkRender::updateProjection()
{
  projViewMat=projMat*viewMat;
}

void AsyVkRender::frustum(double left, double right, double bottom,
                          double top, double nearVal, double farVal)
{
  projMat = glm::frustum(left, right, bottom, top, nearVal, farVal);
  updateProjection();
}

void AsyVkRender::ortho(double left, double right, double bottom,
                        double top, double nearVal, double farVal)
{
  projMat = glm::ortho(left, right, bottom, top, nearVal, farVal);
  updateProjection();
}

#endif

void AsyVkRender::clearCenters()
{
  camp::drawElement::centers.clear();
  camp::drawElement::centermap.clear();
}

void AsyVkRender::clearMaterials()
{
  materials.clear();
  materialMap.clear();
}

#ifdef HAVE_VULKAN

void AsyVkRender::animate()
{
  Animate=!Animate;
  if(Animate) {
    if(Fitscreen == 2) {
      toggleFitScreen();
      toggleFitScreen();
    }
    update();
  } else idle();
}

void AsyVkRender::expand()
{
  double resizeStep=settings::getSetting<double>("resizestep");
  if(resizeStep > 0.0)
    setsize((int) (width*resizeStep+0.5),(int) (height*resizeStep+0.5));
}

void AsyVkRender::shrink()
{
  double resizeStep=settings::getSetting<double>("resizestep");
  if(resizeStep > 0.0)
    setsize(max((int) (width/resizeStep+0.5),1),
            max((int) (height/resizeStep+0.5),1));
}

projection AsyVkRender::camera(bool user)
{
  if(!vkinitialize) return projection();

  camp::Triple vCamera,vUp,vTarget;

  double cz=0.5*(Zmin+Zmax);

  double *Rotate=value_ptr(rotateMat);

  if(user) {
    for(int i=0; i < 3; ++i) {
      double sumCamera=0.0, sumTarget=0.0, sumUp=0.0;
      int i4=4*i;
      for(int j=0; j < 4; ++j) {
        int j4=4*j;
        double R0=Rotate[j4];
        double R1=Rotate[j4+1];
        double R2=Rotate[j4+2];
        double R3=Rotate[j4+3];
        double T4ij=T[i4+j];
        sumCamera += T4ij*(R3-cx*R0-cy*R1-cz*R2);
        sumUp += Tup[i4+j]*R1;
        sumTarget += T4ij*(R3-cx*R0-cy*R1);
      }
      vCamera[i]=sumCamera;
      vUp[i]=sumUp;
      vTarget[i]=sumTarget;
    }
  } else {
    for(int i=0; i < 3; ++i) {
      int i4=4*i;
      double R0=Rotate[i4];
      double R1=Rotate[i4+1];
      double R2=Rotate[i4+2];
      double R3=Rotate[i4+3];
      vCamera[i]=R3-cx*R0-cy*R1-cz*R2;
      vUp[i]=R1;
      vTarget[i]=R3-cx*R0-cy*R1;
    }
  }

  return projection(orthographic,vCamera,vUp,vTarget,Zoom,
                    2.0*atan(tan(0.5*Angle)/Zoom)/radians,
                    pair(X/width+Shift.getx(),
                         Y/height+Shift.gety()));
}

void AsyVkRender::exportHandler(int) {

  vk->readyAfterExport=true;
  vk->Export(0);
}

void AsyVkRender::Export(int imageIndex) {
  exportCommandBuffer->reset();

  vkutils::checkVkResult(device->resetFences(1, &*exportFence));

  exportCommandBuffer->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

  auto const size = device->getImageMemoryRequirements(backbufferImages[0]).size;
  auto const swapExtent = vk::Extent3D(
    backbufferExtent.width,
    backbufferExtent.height,
    1
  );
  auto const reg = vk::BufferImageCopy(
    0,
    backbufferExtent.width,
    backbufferExtent.height,
    vk::ImageSubresourceLayers(
      vk::ImageAspectFlagBits::eColor, 0, 0, 1
    ),
    { },
    swapExtent
  );

  vma::cxx::UniqueBuffer exportBuf = createBufferUnique(
    vk::BufferUsageFlagBits::eTransferDst,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    size,
    VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);

  transitionImageLayout(
    *exportCommandBuffer,
    backbufferImages[imageIndex],
    vk::AccessFlagBits::eMemoryRead,
    vk::AccessFlagBits::eTransferRead,
    !View ? vk::ImageLayout::eColorAttachmentOptimal : vk::ImageLayout::ePresentSrcKHR,
    vk::ImageLayout::eTransferSrcOptimal,
    vk::PipelineStageFlagBits::eTransfer,
    vk::PipelineStageFlagBits::eTransfer,
    vk::ImageSubresourceRange(
      vk::ImageAspectFlagBits::eColor,
      0,
      1,
      0,
      1
    )
  );

  exportCommandBuffer->copyImageToBuffer(backbufferImages[imageIndex], vk::ImageLayout::eTransferSrcOptimal, exportBuf.getBuffer(), 1, &reg);

  transitionImageLayout(
    *exportCommandBuffer,
    backbufferImages[imageIndex],
    vk::AccessFlagBits::eTransferRead,
    vk::AccessFlagBits::eMemoryRead,
    vk::ImageLayout::eTransferSrcOptimal,
    !View ? vk::ImageLayout::eColorAttachmentOptimal : vk::ImageLayout::ePresentSrcKHR,
    vk::PipelineStageFlagBits::eTransfer,
    vk::PipelineStageFlagBits::eTransfer,
    vk::ImageSubresourceRange(
      vk::ImageAspectFlagBits::eColor,
      0,
      1,
      0,
      1
    )
  );

  exportCommandBuffer->end();

  auto const submitInfo = vk::SubmitInfo(
    0, nullptr, nullptr,
    1, &*exportCommandBuffer,
    0, nullptr
  );

  if (renderQueue.submit(1, &submitInfo, *exportFence) != vk::Result::eSuccess)
    runtimeError("failed to submit draw command buffer");

  vkutils::checkVkResult(device->waitForFences(
    1, &*exportFence, VK_TRUE, timeout
  ));

  vma::cxx::MemoryMapperLock mappedMemory(exportBuf);

  auto * fmt = new unsigned char[backbufferExtent.width * backbufferExtent.height * 3]; // 3 for RGB

  auto data=mappedMemory.getCopyPtr<unsigned char>();
  for (auto i = 0u; i < backbufferExtent.height; i++)
    for (auto j = 0u; j < backbufferExtent.width; j++)
      for (auto k = 0u; k < 3; k++)
        // need to flip vertically and swap byte order due to little endian in image data
        // 4 for sizeof unsigned (RGBA)
        fmt[(backbufferExtent.height-1-i)*backbufferExtent.width*3+j*3+(2-k)]=data[i*backbufferExtent.width*4+j*4+k];

  picture pic;
  double w=oWidth;
  double h=oHeight;

  if(settings::verbose > 1)
    cout << "Exporting " << Prefix << " as " << fullWidth << "x"
         << fullHeight << " image" << endl;

  auto * const Image=new camp::drawRawImage(fmt,
                                            backbufferExtent.width,
                                            backbufferExtent.height,
                                            transform(0.0,0.0,w,0.0,0.0,h),
                                            antialias);
  pic.append(Image);
  pic.shipout(NULL,Prefix,Format,false,ViewExport);
  delete Image;
  delete[] fmt;
  queueExport=false;
  setProjection();
  remesh=true;
  redraw=true;

#ifdef HAVE_PTHREAD
  if(vkthread && readyAfterExport) {
    readyAfterExport=false;
    endwait(readySignal,readyLock);
  }
#endif
}

void AsyVkRender::quit()
{
#ifdef HAVE_VULKAN
  resize=false;
  if(vkthread) {
    bool animating=settings::getSetting<bool>("animating");
    if(animating)
      settings::Setting("interrupt")=true;
    redraw=false;
    waitEvent=false;
    Animate=settings::getSetting<bool>("autoplay");
#ifdef HAVE_PTHREAD
    if(!interact::interactive || animating) {
      idle();
      endwait(readySignal,readyLock);
    }

#endif
    if(View) {
      glfwHideWindow(window);
      hideWindow=true;
    }
  } else {
    if(View) {
       if(window) {
        glfwDestroyWindow(window);
        window=nullptr;
      }
      glfwTerminate();
    }
    glslang::FinalizeProcess();

    exit(0);
  }
#endif
}

#ifdef HAVE_VULKAN

void closeWindowHandler(GLFWwindow *window)
{
  cout << endl;
  exitHandler(0);
}

void AsyVkRender::idleFunc(std::function<void()> f)
{
  spinTimer.reset();
  currentIdleFunc = f;
}

void AsyVkRender::idle()
{
  idleFunc(nullptr);
  Xspin=Yspin=Zspin=Animate=Step=false;
}

double AsyVkRender::spinStep()
{
  return settings::getSetting<double>("spinstep")*spinTimer.seconds(true);
}

void AsyVkRender::rotateX(double step)
{
  dmat4 tmpRot(1.0);
  tmpRot=rotate(tmpRot,glm::radians(step),dvec3(1,0,0));
  rotateMat=tmpRot*rotateMat;

  update();
}

void AsyVkRender::rotateY(double step)
{
  dmat4 tmpRot(1.0);
  tmpRot=rotate(tmpRot,glm::radians(step),dvec3(0,1,0));
  rotateMat=tmpRot*rotateMat;

  update();
}

void AsyVkRender::rotateZ(double step)
{
  dmat4 tmpRot(1.0);
  tmpRot=rotate(tmpRot,glm::radians(step),dvec3(0,0,1));
  rotateMat=tmpRot*rotateMat;

  update();
}

void AsyVkRender::xspin()
{
  rotateX(spinStep());
}

void AsyVkRender::yspin()
{
  rotateY(spinStep());
}

void AsyVkRender::zspin()
{
  rotateZ(spinStep());
}

void AsyVkRender::spinx()
{
  if(Xspin)
    idle();
  else {
    idleFunc([this](){xspin();});
    Xspin=true;
    Yspin=Zspin=false;
  }
}

void AsyVkRender::spiny()
{
  if(Yspin)
    idle();
  else {
    idleFunc([this](){yspin();});
    Yspin=true;
    Xspin=Zspin=false;
  }
}

void AsyVkRender::spinz()
{
  if(Zspin)
    idle();
  else {
    idleFunc([this](){zspin();});
    Zspin=true;
    Xspin=Yspin=false;
  }
}

void AsyVkRender::showCamera()
{
  projection P=camera();
  string projection=P.orthographic ? "orthographic(" : "perspective(";
  string indent(2+projection.length(),' ');
  cout << endl
       << "currentprojection=" << endl << "  "
       << projection << "camera=" << P.camera << "," << endl
       << indent << "up=" << P.up << "," << endl
       << indent << "target=" << P.target << "," << endl
       << indent << "zoom=" << P.zoom;
  if(!orthographic)
    cout << "," << endl << indent << "angle=" << P.angle;
  if(P.viewportshift != pair(0.0,0.0))
    cout << "," << endl << indent << "viewportshift=" << P.viewportshift*Zoom;
  if(!orthographic)
    cout << "," << endl << indent << "autoadjust=false";
  cout << ");" << endl;
}

void AsyVkRender::shift(double dx, double dy)
{
  double Zoominv=1.0/Zoom;

  X += dx*Zoominv;
  Y += -dy*Zoominv;
  update();
}

void AsyVkRender::pan(double dx, double dy)
{
  if(orthographic)
    shift(dx,dy);
  else {
    cx += dx * (xmax - xmin) / width;
    cy -= dy * (ymax - ymin) / height;
    update();
  }
}

void AsyVkRender::capzoom()
{
  static double maxzoom=sqrt(DBL_MAX);
  static double minzoom=1.0/maxzoom;
  if(Zoom <= minzoom) Zoom=minzoom;
  if(Zoom >= maxzoom) Zoom=maxzoom;

  if(fabs(Zoom-lastzoom) > settings::getSetting<double>("zoomThreshold")) {
    remesh=true;
    lastzoom=Zoom;
  }
}

void AsyVkRender::zoom(double dx, double dy)
{
  double zoomFactor=settings::getSetting<double>("zoomfactor");
  if (zoomFactor > 0.0) {
    double zoomStep=settings::getSetting<double>("zoomstep");
    const double limit=log(0.1*DBL_MAX)/log(zoomFactor);
    double stepPower=zoomStep*dy;
    if(fabs(stepPower) < limit) {
      Zoom *= std::pow(zoomFactor,-stepPower);
      update();
    }
  }
}

void AsyVkRender::capsize(int& width, int& height) {

  if(width > screenWidth)
    width=screenWidth;
  if(height > screenHeight)
    height=screenHeight;
}

void AsyVkRender::windowposition(int& x, int& y, int Width, int Height)
{
  if (width == -1) {
    Width=width;
  }
  if (height == -1) {
    Height=height;
  }

  pair z=settings::getSetting<pair>("position");
  x=(int) z.getx();
  y=(int) z.gety();
  if(x < 0) {
    x += screenWidth-width;
    if(x < 0) x=0;
  }
  if(y < 0) {
    y += screenHeight-height;
    if(y < 0) y=0;
  }
}

void AsyVkRender::setsize(int w, int h, bool reposition) {
  int x,y;
  capsize(w,h);

  if (View) {
    glfwSetWindowSize(window, w, h);

    if (reposition) {
      windowposition(x, y, w, h);
      glfwSetWindowPos(window, x, y);
    }
  }

  reshape0(w,h);
  update();
}

void AsyVkRender::fullscreen(bool reposition)
{
  width=screenWidth;
  height=screenHeight;
  Xfactor=((double) screenHeight)/height;
  Yfactor=((double) screenWidth)/width;
  if(reposition)
    glfwSetWindowPos(window, 0, 0);
  setsize(width,height,reposition);
}

void AsyVkRender::reshape0(int Width, int Height) {
  X=(X/width)*Width;
  Y=(Y/height)*Height;

  width=Width;
  height=Height;

  static int lastWidth=1;
  static int lastHeight=1;
  if(View && width*height > 1 &&
     (width != lastWidth || height != lastHeight)) {

    if(settings::verbose > 1)
      cout << "Rendering " << stripDir(Prefix) << " as "
           << width << "x" << height << " image" << endl;
    lastWidth=width;
    lastHeight=height;
  }

  setProjection();
  framebufferResized=true;
}

void AsyVkRender::setosize() {
  oldWidth=(int) ceil(oWidth);
  oldHeight=(int) ceil(oHeight);
}

void AsyVkRender::fitscreen(bool reposition) {
  if(Animate && Fitscreen == 2) Fitscreen=0;

  switch(Fitscreen) {
    case 0: // Original size
    {
      Xfactor=Yfactor=1.0;
      double pixelRatio=settings::getSetting<double>("devicepixelratio");
      setsize(oldWidth*pixelRatio,oldHeight*pixelRatio,reposition);
      break;
    }
    case 1: // Fit to screen in one dimension
    {
      int w=screenWidth;
      int h=screenHeight;
      if(w > h*Aspect)
        w=min((int) ceil(h*Aspect),w);
      else
        h=min((int) ceil(w/Aspect),h);

      setsize(w,h,reposition);
      break;
    }
    case 2: // Full screen
    {
      fullscreen(reposition);
      break;
    }
  }
}

void AsyVkRender::toggleFitScreen() {
  glfwHideWindow(window);
  Fitscreen = (Fitscreen + 1) % 3;
  fitscreen();
}

void AsyVkRender::home(bool webgl) {
  if(!webgl)
    idle();
  X = Y = cx = cy = 0;
  rotateMat = viewMat = dmat4(1.0);
  lastzoom=Zoom=Zoom0;
  framecount=0;

  setProjection();
  updateModelViewData();
}

void AsyVkRender::cycleMode() {
  if(device)
    device->waitIdle();
  mode=DrawMode((mode + 1) % DRAWMODE_MAX);
  remesh=true;
  redraw=true;
  newUniformBuffer=true;

  if (mode == DRAWMODE_NORMAL) {
    ibl=settings::getSetting<bool>("ibl");
  }
  if (mode == DRAWMODE_OUTLINE) {
    ibl=false;
  }
  recreatePipeline=true;
}

#endif

#endif

} // namespace camp

#endif // HAVE_LIBGLM
