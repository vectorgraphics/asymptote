#define VMA_IMPLEMENTATION
#include "vkrender.h"
#include "shaderResources.h"
#include "picture.h"
#include "drawimage.h"
#include "EXRFiles.h"

#define SHADER_DIRECTORY "base/shaders/"
#define VALIDATION_LAYER "VK_LAYER_KHRONOS_validation"
#define MESA_OVERLAY_LAYER "VK_LAYER_MESA_overlay"

//using namespace settings;

bool havewindow;

static bool initialized=false;

void exitHandler(int);

#ifdef HAVE_VULKAN
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE;

std::vector<const char*> instanceExtensions
{
  VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
  VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME,
#ifdef VALIDATION
  VK_EXT_DEBUG_UTILS_EXTENSION_NAME
#endif
};
#endif

#ifdef HAVE_LIBGLM

namespace camp
{

static bool vkinitialize=true;

std::vector<char> readFile(const std::string& filename)
{
  std::ifstream file(filename, std::ios::ate | std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("failed to open file " + filename + "!");
  }

  std::size_t fileSize = (std::size_t) file.tellg();
  std::vector<char> buffer(fileSize);

  file.seekg(0);
  file.read(buffer.data(), fileSize);
  file.close();

  return buffer;
}

#ifdef HAVE_VULKAN
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
    if (availableFormat.format == vk::Format::eB8G8R8A8Uint &&
        availableFormat.colorSpace == vk::ColorSpaceKHR::eAdobergbLinearEXT) {
      return availableFormat;
    }
  }

  return formats.front();
}

vk::PresentModeKHR
SwapChainDetails::choosePresentMode() const
{
  for (const auto& mode : presentModes) {
    if (mode == vk::PresentModeKHR::eImmediate) {
      return mode;
    }
  }

  return presentModes.front();
}

vk::Extent2D
SwapChainDetails::chooseExtent() const
{
  if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
    return capabilities.currentExtent;
  }

  int width, height;
  glfwGetFramebufferSize(vk->window, &width, &height);

  auto extent = vk::Extent2D(
    static_cast<uint32_t>(width),
    static_cast<uint32_t>(height)
  );

  extent.width = glm::clamp(
                  extent.width,
                  capabilities.minImageExtent.width,
                  capabilities.maxImageExtent.width
                 );
  extent.height = glm::clamp(
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
  double xshift = (x / (double) width + Shift.getx() * Xfactor) * Zoom0;
  double yshift = (y / (double) height + Shift.gety() * Yfactor) * Zoom0;
  double zoominv = 1.0 / Zoom0;
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

  if (orthographic) {
    projMat = glm::ortho(xmin, xmax, ymin, ymax, -Zmax, -Zmin);
  } else {
    projMat = glm::frustum(xmin, xmax, ymin, ymax, -Zmax, -Zmin);
  }

  newUniformBuffer = true;
}

void AsyVkRender::updateViewmodelData()
{
  normMat = glm::inverse(viewMat);
  double *T=glm::value_ptr(normMat);
  for(size_t i=0; i < 9; ++i)
    BBT[i]=T[i];
}

void AsyVkRender::update()
{
  capzoom();

  double cz = 0.5 * (Zmin + Zmax);
  viewMat = glm::translate(glm::translate(glm::dmat4(1.0), glm::dvec3(cx, cy, cz)) * rotateMat, glm::dvec3(0, 0, -cz));

  setProjection();
  updateViewmodelData();

  static auto const verticalFlipMat = glm::scale(glm::dmat4(1.0f), glm::dvec3(1.0f, -1.0f, 1.0f));

  projViewMat = verticalFlipMat * projMat * viewMat;
  redraw=true;
}

triple AsyVkRender::billboardTransform(const triple& center, const triple& v) const
{
  double cx = center.getx();
  double cy = center.gety();
  double cz = center.getz();

  double x = v.getx() - cx;
  double y = v.gety() - cy;
  double z = v.getz() - cz;

  const double* BBT = glm::value_ptr(normMat);

  return triple(x * BBT[0] + y * BBT[3] + z * BBT[6] + cx,
                x * BBT[1] + y * BBT[4] + z * BBT[7] + cy,
                x * BBT[2] + y * BBT[5] + z * BBT[8] + cz);
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
  triple B(Xmax, Ymin, Zmax);
  pair size3(s * (B.getx() - b.getx()), s * (B.gety() - b.gety()));
  pair size2(width, height);
  return prerender * size3.length() / size2.length();
}

#ifdef HAVE_VULKAN

void AsyVkRender::initWindow()
{
  if (!View)
    return;

  if (!window) {

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window = glfwCreateWindow(width, height, title.data(), nullptr, nullptr);
  }

  glfwShowWindow(window);
  glfwSetWindowUserPointer(window, this);
  glfwSetMouseButtonCallback(window, mouseButtonCallback);
  glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
  glfwSetScrollCallback(window, scrollCallback);
  glfwSetCursorPosCallback(window, cursorPosCallback);
  glfwSetKeyCallback(window, keyCallback);

  // call this to set thread signal behavior
  framebufferResizeCallback(window, width, height);
}

void AsyVkRender::updateHandler(int) {

  vk->queueScreen=true;
  vk->remesh=true;
  vk->redraw=true;
  vk->update();
  if(interact::interactive || !vk->Animate) {
    glfwShowWindow(vk->window);
  }
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

void AsyVkRender::mouseButtonCallback(GLFWwindow * window, int button, int action, int mods)
{
  auto const currentAction = getAction(button, mods);

  if (currentAction.empty())
    return;

  auto app = reinterpret_cast<AsyVkRender*>(glfwGetWindowUserPointer(window));

  app->lastAction = currentAction;
}

void AsyVkRender::framebufferResizeCallback(GLFWwindow* window, int width, int height)
{
  auto app = reinterpret_cast<AsyVkRender*>(glfwGetWindowUserPointer(window));
  app->Y = (app->X / app->width) * width;
  app->Y = (app->Y / app->height) * height;
  app->width = width;
  app->height = height;
  app->fullWidth = width;
  app->fullHeight = height;
  app->framebufferResized = true;
  app->update();

  if(app->vkthread) {
    static bool initialize=true;
    if(initialize) {
      initialize=false;
      Signal(SIGUSR1,updateHandler);
    }
  }
}

void AsyVkRender::scrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
  auto app = reinterpret_cast<AsyVkRender*>(glfwGetWindowUserPointer(window));
  auto zoomFactor = settings::getSetting<double>("zoomfactor");

  if (zoomFactor == 0.0)
    return;

  if (yoffset > 0)
    app->Zoom0 *= zoomFactor;
  else
    app->Zoom0 /= zoomFactor;

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
    app->rotateMat = glm::rotate(2 * arcball.angle / app->Zoom0 * app->ArcballFactor,
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

void AsyVkRender::keyCallback(GLFWwindow * window, int key, int scancode, int action, int mods)
{
  if (action != GLFW_PRESS)
    return;

  auto app = reinterpret_cast<AsyVkRender*>(glfwGetWindowUserPointer(window));

  switch (key)
  {
    case 'H':
      app->travelHome();
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
    case 'p':
      if(settings::getSetting<bool>("reverse")) app->Animate=false;
      settings::Setting("reverse")=app->Step=false;
      app->animate();
      break;
    case 'r':
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
      if(!app->Format.empty()) app->Export(0);
      app->quit();
      break;
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

void AsyVkRender::vkrender(const string& prefix, const picture* pic, const string& format,
                           double Width, double Height, double angle, double zoom,
                           const triple& mins, const triple& maxs, const pair& shift,
                           const pair& margin, double* t,
                           double* background, size_t nlightsin, triple* lights,
                           double* diffuse, double* specular, bool view, int oldpid/*=0*/)
{
  // By default use discrete GPU.
  setenv("DRI_PRIME","1",0);

  glfwInit();

  offscreen=settings::getSetting<bool>("offscreen");
  GLFWmonitor* monitor=glfwGetPrimaryMonitor();
  if(!monitor) offscreen=true;

  this->pic = pic;
  this->Prefix=prefix;
  this->Format = format;
  this->shouldUpdateBuffers = true;
  this->redraw = true;
  this->remesh = true;
  this->nlights = nlightsin;
  this->Lights = lights;
  this->LightsDiffuse = diffuse;
  this->Oldpid = oldpid;

  this->Angle = angle * radians;
  this->Zoom0 = zoom;
  this->Shift = shift / zoom;
  this->Margin = margin;

  for (int i = 0; i < 4; i++)
    this->Background[i] = static_cast<float>(background[i]);

  this->ViewExport=view;
  this->View = view && !offscreen;

  this->title = std::string(settings::PROGRAM)+": "+prefix.c_str();

  Xmin = mins.getx();
  Xmax = maxs.getx();
  Ymin = mins.gety();
  Ymax = maxs.gety();
  Zmin = mins.getz();
  Zmax = maxs.getz();

  orthographic = (this->Angle == 0.0);
  H = orthographic ? 0.0 : -tan(0.5 * this->Angle) * Zmax;
  Xfactor = Yfactor = 1.0;

  bool v3d=format == "v3d";
  bool webgl=format == "html";
  bool format3d=webgl || v3d;

#ifdef HAVE_PTHREAD
  static bool initializedView=false;
  if(vkinitialize)
    Fitscreen=1;
#endif

  for(int i=0; i < 16; ++i)
    T[i]=t[i];

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

    fullWidth=(int) ceil(expand*Width);
    fullHeight=(int) ceil(expand*Height);

    if(offscreen) {
      screenWidth=fullWidth;
      screenHeight=fullHeight;
    } else {
      int mx, my;
      glfwGetMonitorWorkarea(monitor, &mx, &my, &screenWidth, &screenHeight);
    }

    oWidth=Width;
    oHeight=Height;
    Aspect=Width/Height;

    if(format3d) {
      width=fullWidth;
      height=fullHeight;
    } else {
      width=min(fullWidth,screenWidth);
      height=min(fullHeight,screenHeight);

      if(width > height*Aspect)
        width=min((int) (ceil(height*Aspect)),screenWidth);
      else
        height=min((int) (ceil(width/Aspect)),screenHeight);
    }

    travelHome(format3d);
    setProjection();
    if(format3d) {
      remesh=true;
      return;
    }

    maxFragments=0;

    rotateMat = glm::mat4(1.0);
    viewMat = glm::mat4(1.0);
    ArcballFactor=1+8.0*hypot(Margin.getx(),Margin.gety())/hypot(width,height);

#ifdef HAVE_VULKAN
    Aspect=((double) width)/height; // CHECK
    setosize();
#endif
  }

#ifdef HAVE_VULKAN
  havewindow=initialized && vkthread;

  clearMaterials();
  initialized=true;
#endif

#ifdef HAVE_PTHREAD
  if(vkthread && initializedView) {
    if(View) {
#ifdef __MSDOS__ // Signals are unreliable in MSWindows
      vkupdate=true;
#else
      pthread_kill(mainthread,SIGUSR1);
#endif
    } else readyAfterExport=queueExport=true;
    return;
  }
#endif

  GPUindexing=settings::getSetting<bool>("GPUindexing");
  GPUcompress=GPUindexing && settings::getSetting<bool>("GPUcompress");

  if(GPUindexing) {
    localSize=settings::getSetting<Int>("GPUlocalSize");
    blockSize=settings::getSetting<Int>("GPUblockSize");
    groupSize=localSize*blockSize;
  }

#ifdef HAVE_VULKAN
  if(vkinitialize) {
    vkinitialize=false;

  interlock=settings::getSetting<bool>("GPUinterlock");

#ifdef HAVE_VULKAN
    Aspect=((double) width)/height;
    setosize();
#endif

  Animate=settings::getSetting<bool>("autoplay") && vkthread;
  ibl=settings::getSetting<bool>("ibl");

  if (offscreen && settings::verbose > 1) {
    std::cout << "Using offscreen mode." << std::endl;
  }

    travelHome(format3d);
    setProjection();
  initWindow();

  if(View) {
    if(!settings::getSetting<bool>("fitscreen"))
      Fitscreen=0;
    firstFit=true;
    fitscreen();
    setosize();
  }

  initVulkan();
  }


  if(View) {
    if(!settings::getSetting<bool>("fitscreen"))
      Fitscreen=0;
    firstFit=true;
    fitscreen();
    setosize();
    initializedView=true;
  }

  update();

#ifdef __MSDOS__
//  if(vkthread && interact::interactive)
//    poll(0);
#endif

  mainLoop();

#endif
}

#ifdef HAVE_VULKAN
void AsyVkRender::initVulkan()
{
  VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

  if (!glslang::InitializeProcess()) {
    throw std::runtime_error("Unable to initialize glslang.");
  }

  maxFramesInFlight=View ? settings::getSetting<Int>("maxFramesInFlight") : 1;
  frameObjects.resize(maxFramesInFlight);

  if (settings::verbose > 1) {
    std::cout << "Using " << maxFramesInFlight
              << " maximum frame(s) in flight." << std::endl;
  }

  createInstance();
  if (View) createSurface();
  pickPhysicalDevice();
  createLogicalDevice();
  createAllocator();
  createCommandPools();
  createCommandBuffers();
  if (View) createSwapChain();
  else createOffscreenBuffers();
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
  writeDescriptorSets();
  writeMaterialAndLightDescriptors();

  createCountRenderPass();
  createGraphicsRenderPass();
  createGraphicsPipelineLayout();
  createGraphicsPipelines();
  if (!Opaque && GPUindexing) {
    createComputePipelines();
  }

  createAttachments();

  createFramebuffers();
  createExportResources();
}

void AsyVkRender::recreateSwapChain()
{
  int width = 0, height = 0;
  glfwGetFramebufferSize(window, &width, &height);

  while (width == 0 || height == 0)
  {
    glfwGetFramebufferSize(window, &width, &height);
    glfwWaitEvents();
  }

//  device->waitIdle();

  createSwapChain();
  if(!Opaque) createDependentBuffers();
  writeDescriptorSets();
  createImageViews();
  if(!Opaque)
    createCountRenderPass();
  createGraphicsRenderPass();
  createGraphicsPipelines();
  createAttachments();
  createFramebuffers();
  createExportResources();

  redraw=true;
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
    VK_API_VERSION_1_2
  );
  auto extensions = getRequiredInstanceExtensions();
  auto supportedLayers = vk::enumerateInstanceLayerProperties();

  auto isLayerSupported = [supportedLayers](std::string layerName) {
    return std::find_if(
      supportedLayers.begin(),
      supportedLayers.end(),
      [layerName](vk::LayerProperties const& layer) {
        return layer.layerName.data() == layerName;
      }) != supportedLayers.end();
  };

#ifdef VALIDATION
  if (isLayerSupported(VALIDATION_LAYER)) {
    validationLayers.emplace_back(VALIDATION_LAYER);
  } else if (settings::verbose > 1) {
    std::cout << "Validation layers are not supported by the current Vulkan instance." << std::endl;
  }
#endif

  if (settings::verbose > 2) {
    if (isLayerSupported(MESA_OVERLAY_LAYER)) {
      validationLayers.emplace_back(MESA_OVERLAY_LAYER);
    } else if (settings::verbose > 1) {
      std::cout << "Mesa overlay layer is not supported by the current Vulkan instance." << std::endl;
    }
  }

  auto const instanceCI = vk::InstanceCreateInfo(
    vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR,
    &appInfo,
    validationLayers.size(),
    validationLayers.data(),
    extensions.size(),
    extensions.data()
  );
  instance = vk::createInstanceUnique(instanceCI);
  VULKAN_HPP_DEFAULT_DISPATCHER.init(*instance);
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
  if (glfwCreateWindowSurface(*instance, window, nullptr, &surfaceTmp) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create window surface!");
  }
  surface = vk::UniqueSurfaceKHR(surfaceTmp, *instance);
#endif
}

void AsyVkRender::createAllocator()
{
  VmaVulkanFunctions vkFuncs = {};
  vkFuncs.vkGetInstanceProcAddr = vk::defaultDispatchLoaderDynamic.vkGetInstanceProcAddr;
  vkFuncs.vkGetDeviceProcAddr = vk::defaultDispatchLoaderDynamic.vkGetDeviceProcAddr;

  VmaAllocatorCreateInfo createInfo = {};
  createInfo.vulkanApiVersion = VK_API_VERSION_1_2;
  createInfo.physicalDevice = physicalDevice;
  createInfo.device = *device;
  createInfo.instance = *instance;
  createInfo.pVulkanFunctions = &vkFuncs;

  allocator = vma::cxx::UniqueAllocator(createInfo);
}

void AsyVkRender::pickPhysicalDevice()
{
  auto const getDeviceScore =
  [this](vk::PhysicalDevice& device) -> std::size_t
  {
    std::size_t score = 0u;

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
      score += 10;
    }
    else if(vk::PhysicalDeviceType::eIntegratedGpu == props.deviceType) {
      score += 5;
    }
    else if (vk::PhysicalDeviceType::eCpu == props.deviceType && offscreen) {
      // Force using cpu for offscreen
      score += 100;
    }

    return score;
  };

  std::pair<std::size_t, vk::PhysicalDevice> highestDeviceScore;

  for (auto & dev: instance->enumeratePhysicalDevices())
  {
    auto const score = getDeviceScore(dev);

    if (nullptr == highestDeviceScore.second
        || score > highestDeviceScore.first)
      highestDeviceScore = std::make_pair(score, dev);
  }

  if (0 == highestDeviceScore.first) {
    throw std::runtime_error("No suitable GPUs.");
  }

  physicalDevice = highestDeviceScore.second;
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

  if (supportedDeviceExtensions.find(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME) != supportedDeviceExtensions.end()) {
    extensions.push_back(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
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

  physicalDevice.getProperties2(&props);

  if (interlock) {
    portabilityFeatures.pNext = &interlockFeatures;
  }

  auto deviceCI = vk::DeviceCreateInfo(
    vk::DeviceCreateFlags(),
    VEC_VIEW(queueCIs),
    VEC_VIEW(validationLayers),
    VEC_VIEW(extensions),
    &deviceFeatures,
    &portabilityFeatures
  );

  device = physicalDevice.createDeviceUnique(deviceCI, nullptr);
  VULKAN_HPP_DEFAULT_DISPATCHER.init(*device);

  transferQueue = device->getQueue(queueFamilyIndices.transferQueueFamily, 0);
  renderQueue = device->getQueue(queueFamilyIndices.renderQueueFamily, 0);
  if (queueFamilyIndices.presentQueueFamilyFound) {
    presentQueue = device->getQueue(queueFamilyIndices.presentQueueFamily, 0);
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
  auto && extent = swapDetails.chooseExtent();

  vk::SwapchainCreateInfoKHR swapchainCI = vk::SwapchainCreateInfoKHR(
    vk::SwapchainCreateFlagsKHR(),
    *surface,
    swapDetails.chooseImageCount(),
    format.format,
    format.colorSpace,
    extent,
    1,
    vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc,
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
  defaultBackbufferImg = createImage(
          backbufferExtent.width,
          backbufferExtent.height,
              vk::SampleCountFlagBits::e1, backbufferImageFormat,
              vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc,
              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  backbufferImages.emplace_back(defaultBackbufferImg.getImage());

  for(auto & image: backbufferImages) {
    transitionImageLayout(vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal, image);
  }
}

void AsyVkRender::createImageViews()
{
  backbufferImageViews.resize(backbufferImages.size());
  for (size_t i = 0; i < backbufferImages.size(); i++) {
    vk::ImageViewCreateInfo viewCI = vk::ImageViewCreateInfo(vk::ImageViewCreateFlags(), backbufferImages[i], vk::ImageViewType::e2D, backbufferImageFormat, vk::ComponentMapping(), vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
    backbufferImageViews[i] = device->createImageViewUnique(viewCI, nullptr);
  }
}


vk::UniqueShaderModule AsyVkRender::createShaderModule(EShLanguage lang, std::string const & filename, std::vector<std::string> const & options)
{
  std::string header = "#version 450\n";

  for (auto const & option: options) {
    header += "#define " + option + "\n";
  }

  auto fileContents = readFile(filename);
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
    std::cout << fileContents.size() << std::endl;
    std::cout << fileContents.data() << std::endl;
    throw std::runtime_error("Failed to parse shader "
                             + filename
                             + ": " + shader.getInfoLog()
                             + " " + shader.getInfoDebugLog());
  }

  program.addShader(&shader);

  if (!program.link(compileMessages)) {
    throw std::runtime_error("Failed to link shader "
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

  for (auto i = 0u; i < backbufferImageViews.size(); i++)
  {
    std::array<vk::ImageView, 3> attachments = {*colorImageView, *depthImageView, *backbufferImageViews[i]};
    std::array<vk::ImageView, 1> depthAttachments {*depthImageView};

    auto depthFramebufferCI = vk::FramebufferCreateInfo(
      {},
      *countRenderPass,
      STD_ARR_VIEW(depthAttachments),
      backbufferExtent.width,
      backbufferExtent.height,
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

    depthFramebuffers[i] = device->createFramebufferUnique(depthFramebufferCI);
    opaqueGraphicsFramebuffers[i] = device->createFramebufferUnique(opaqueGraphicsFramebufferCI);
    graphicsFramebuffers[i] = device->createFramebufferUnique(graphicsFramebufferCI);
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
  cmd.end();

  auto info = vk::SubmitInfo();

  info.commandBufferCount = 1;
  info.pCommandBuffers = &cmd;

  renderQueue.submit(1, &info, *fence); // todo transfer queue
  device->waitForFences(1, &*fence, true, std::numeric_limits<std::uint64_t>::max());

  device->freeCommandBuffers(*renderCommandPool, 1, &cmd);
}

void AsyVkRender::createSyncObjects()
{
  for (auto i = 0; i < maxFramesInFlight; i++) {
    frameObjects[i].imageAvailableSemaphore = device->createSemaphoreUnique(vk::SemaphoreCreateInfo());
    frameObjects[i].renderFinishedSemaphore = device->createSemaphoreUnique(vk::SemaphoreCreateInfo());
    frameObjects[i].inCountBufferCopy = device->createSemaphoreUnique(vk::SemaphoreCreateInfo());
    frameObjects[i].inFlightFence = device->createFenceUnique(vk::FenceCreateInfo(vk::FenceCreateFlagBits::eSignaled));
    frameObjects[i].inComputeFence = device->createFenceUnique(vk::FenceCreateInfo(vk::FenceCreateFlagBits::eSignaled));
    frameObjects[i].compressionFinishedEvent = device->createEventUnique(vk::EventCreateInfo());
    frameObjects[i].sumFinishedEvent = device->createEventUnique(vk::EventCreateInfo());
    frameObjects[i].startTimedSumsEvent = device->createEventUnique(vk::EventCreateInfo());
    frameObjects[i].timedSumsFinishedEvent = device->createEventUnique(vk::EventCreateInfo());
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
  throw std::runtime_error("failed to find suitable memory type!");
}

void AsyVkRender::zeroBuffer(vk::Buffer & buf, vk::DeviceSize size) {

  auto const cmd = beginSingleCommands();

  cmd.fillBuffer(buf, 0, size, 0);

  endSingleCommands(cmd);
}

vma::cxx::UniqueBuffer AsyVkRender::createBufferUnique(
        vk::BufferUsageFlags const& usage,
        VkMemoryPropertyFlags const& properties,
        vk::DeviceSize const& size,
        VmaAllocationCreateFlags const& vmaFlags)
{
  auto bufferCI = vk::BufferCreateInfo(vk::BufferCreateFlags(), size, usage);

  VmaAllocationCreateInfo createInfo = {};
  createInfo.usage = VMA_MEMORY_USAGE_AUTO;
  createInfo.requiredFlags = properties;
  createInfo.flags=vmaFlags;

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
  if (submitResult != vk::Result::eSuccess) throw std::runtime_error("failed to submit command buffer!");
  device->waitForFences(1, &*fence, VK_TRUE, std::numeric_limits<uint64_t>::max());
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

void AsyVkRender::copyToBuffer(
        const vk::Buffer& buffer,
        const void* data,
        vk::DeviceSize size
)
{
  vma::cxx::UniqueBuffer stagingBuffer = createBufferUnique(
          vk::BufferUsageFlagBits::eTransferSrc,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          size,
          VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
          );

  copyToBuffer(buffer, data, size, stagingBuffer);
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
  vma::cxx::UniqueBuffer stagingBf= createBufferUnique(
          vk::BufferUsageFlagBits::eTransferDst,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          size,
          VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
  );

  auto const cmd = beginSingleCommands();
  auto const cpy = vk::BufferCopy(
    0, 0, size
  );

  cmd.copyBuffer(buffer, stagingBf.getBuffer(), 1, &cpy);

  endSingleCommands(cmd);

  vma::cxx::MemoryMapperLock const mappedMem(stagingBf);
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

  vma::cxx::UniqueBuffer stagingBf = createBufferUnique(
          vk::BufferUsageFlagBits::eTransferSrc,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          size,
          VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );

  if (vma::cxx::MemoryMapperLock mappedMem(stagingBf); true)
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

  cmd.copyBufferToImage(stagingBf.getBuffer(), img, vk::ImageLayout::eTransferDstOptimal, 1, &cpy);

  endSingleCommands(cmd);
}

void AsyVkRender::setDeviceBufferData(DeviceBuffer& buffer, const void* data, vk::DeviceSize size, std::size_t nobjects)
{
  // Vulkan doesn't allow a buffer to have a size of 0
  auto bufferCI = vk::BufferCreateInfo(vk::BufferCreateFlags(), std::max(vk::DeviceSize(16), size), buffer.usage);
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
}

void AsyVkRender::writeDescriptorSets()
{
  for (auto i = 0; i < maxFramesInFlight; i++)
  {
    auto uboInfo = vk::DescriptorBufferInfo();

    uboInfo.buffer = frameObjects[i].uboBf.getBuffer();
    uboInfo.offset = 0;
    uboInfo.range = sizeof(UniformBufferObject);

    auto countBufferInfo = vk::DescriptorBufferInfo();

    countBufferInfo.buffer = countBf.getBuffer();
    countBufferInfo.offset = 0;
    countBufferInfo.range = countBufferSize;

    auto offsetBufferInfo = vk::DescriptorBufferInfo();

    offsetBufferInfo.buffer = offsetBf.getBuffer();
    offsetBufferInfo.offset = 0;
    offsetBufferInfo.range = offsetBufferSize;

    auto opaqueBufferInfo = vk::DescriptorBufferInfo();

    opaqueBufferInfo.buffer = opaqueBf.getBuffer();
    opaqueBufferInfo.offset = 0;
    opaqueBufferInfo.range = opaqueBufferSize;

    auto opaqueDepthBufferInfo = vk::DescriptorBufferInfo();

    opaqueDepthBufferInfo.buffer = opaqueDepthBf.getBuffer();
    opaqueDepthBufferInfo.offset = 0;
    opaqueDepthBufferInfo.range = opaqueDepthBufferSize;

    auto indexBufferInfo = vk::DescriptorBufferInfo();

    indexBufferInfo.buffer = indexBf.getBuffer();
    indexBufferInfo.offset = 0;
    indexBufferInfo.range = indexBufferSize;

    auto elementBufferInfo = vk::DescriptorBufferInfo();

    elementBufferInfo.buffer = elementBf.getBuffer();
    elementBufferInfo.offset = 0;
    elementBufferInfo.range = elementBufferSize;

    std::array<vk::WriteDescriptorSet, 7> writes;

    writes[0].dstSet = *frameObjects[i].descriptorSet;
    writes[0].dstBinding = 0;
    writes[0].dstArrayElement = 0;
    writes[0].descriptorType = vk::DescriptorType::eUniformBuffer;
    writes[0].descriptorCount = 1;
    writes[0].pBufferInfo = &uboInfo;

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

    device->updateDescriptorSets(GPUcompress ? writes.size() : writes.size()-2,
                                 writes.data(), 0, nullptr);

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

  fragmentBufferSize = maxFragments*sizeof(glm::vec4);
  fragmentBf = createBufferUnique(
          vk::BufferUsageFlagBits::eStorageBuffer,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
          fragmentBufferSize);

  depthBufferSize = maxFragments*sizeof(float);
  depthBf = createBufferUnique(
          vk::BufferUsageFlagBits::eStorageBuffer,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
          depthBufferSize);

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

  // if the fragment buffer size changes, all
  // transparent data needs to be re-rendered
  // for every frame
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
    VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
  );

  if(GPUcompress) {
    elementBf = createBufferUnique(
      vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
      elementBufferSize,
      VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
      );
  }

  for (auto& frameObj : frameObjects)
  {
    frameObj.uboBf = createBufferUnique(
      vk::BufferUsageFlagBits::eUniformBuffer,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      sizeof(UniformBufferObject),
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
      );
    frameObj.uboMappedMemory = make_unique<vma::cxx::MemoryMapperLock>(frameObj.uboBf);
  }

  createMaterialAndLightBuffers();
  createDependentBuffers();
}


void AsyVkRender::createMaterialAndLightBuffers() {
  materialBf = createBufferUnique(
                      vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                      sizeof(camp::Material) * nmaterials);

  lightBf = createBufferUnique(
                     vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     sizeof(camp::Light) * nlights);
}

void AsyVkRender::createDependentBuffers()
{
  pixels=(backbufferExtent.width+1)*(backbufferExtent.height+1);
  std::uint32_t Pixels;

  if (GPUindexing) {
    std::uint32_t G=ceilquotient(pixels,groupSize);
    Pixels=groupSize*G;
    globalSize=localSize*ceilquotient(G,localSize)*sizeof(std::uint32_t);
  }
  else {
    Pixels=pixels;
    globalSize=1;
  }

  countBufferSize=(Pixels+2)*sizeof(std::uint32_t);
  offsetBufferSize=(Pixels+2)*sizeof(std::uint32_t);
  opaqueBufferSize=pixels*sizeof(glm::vec4);
  opaqueDepthBufferSize=sizeof(std::uint32_t)+pixels*sizeof(float);
  indexBufferSize=pixels*sizeof(std::uint32_t);

  VkMemoryPropertyFlags countBufferFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  VmaAllocationCreateFlags vmaFlags = 0;

  if (!GPUindexing) {
    countBufferFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    vmaFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
  }

  if (countBfMappedMem != nullptr)
  {
    countBfMappedMem = nullptr;
  }

  if (offsetStageBfMappedMem != nullptr)
  {
    offsetStageBfMappedMem = nullptr;
  }

  countBf = createBufferUnique(
          vk::BufferUsageFlagBits::eStorageBuffer
                  | vk::BufferUsageFlagBits::eTransferDst
                  | vk::BufferUsageFlagBits::eTransferSrc,
          countBufferFlags,
          countBufferSize,
          vmaFlags
          );

  globalSumBf = createBufferUnique(
          vk::BufferUsageFlagBits::eStorageBuffer,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
          globalSize);

  offsetBf = createBufferUnique(
          vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
          offsetBufferSize);

  if (!GPUindexing) {
    offsetStageBf = createBufferUnique(
      vk::BufferUsageFlagBits::eTransferSrc,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
      offsetBufferSize,
      VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
    );
  }
  opaqueBf = createBufferUnique(
                     vk::BufferUsageFlagBits::eStorageBuffer,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     opaqueBufferSize);

  opaqueDepthBf = createBufferUnique(
                     vk::BufferUsageFlagBits::eStorageBuffer,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     opaqueBufferSize);

  if(GPUcompress) {
    indexBf = createBufferUnique(
      vk::BufferUsageFlagBits::eStorageBuffer,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      indexBufferSize,
      VMA_MEMORY_USAGE_GPU_ONLY);
  }

  if (!GPUindexing) {
    countBfMappedMem = make_unique<vma::cxx::MemoryMapperLock>(countBf);
    offsetStageBfMappedMem = make_unique<vma::cxx::MemoryMapperLock>(offsetStageBf);
  }
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
                      sizeof(glm::vec4) * w * h,
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

  std::array<vk::AttachmentDescription2, 1> attachments {depthAttachment};

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
    attachments.size(),
    attachments.data(),
    subpasses.size(),
    subpasses.data(),
    dependencies.size(),
    dependencies.data()
  );

  countRenderPass = device->createRenderPass2Unique(renderPassCI);
}

void AsyVkRender::createGraphicsRenderPass()
{
  auto colorAttachment = vk::AttachmentDescription2(
    vk::AttachmentDescriptionFlags(),
    backbufferImageFormat,
    msaaSamples,
    vk::AttachmentLoadOp::eClear,
    vk::AttachmentStoreOp::eStore,
    vk::AttachmentLoadOp::eDontCare,
    vk::AttachmentStoreOp::eDontCare,
    vk::ImageLayout::eUndefined,
    vk::ImageLayout::eColorAttachmentOptimal
  );
  auto colorResolveAttachment = vk::AttachmentDescription2(
    vk::AttachmentDescriptionFlags(),
    backbufferImageFormat,
    vk::SampleCountFlagBits::e1,
    vk::AttachmentLoadOp::eDontCare,
    vk::AttachmentStoreOp::eStore,
    vk::AttachmentLoadOp::eDontCare,
    vk::AttachmentStoreOp::eDontCare,
    vk::ImageLayout::eUndefined,
    !View ? vk::ImageLayout::eColorAttachmentOptimal : vk::ImageLayout::ePresentSrcKHR
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
  auto colorResolveAttachmentRef = vk::AttachmentReference2(2, vk::ImageLayout::eColorAttachmentOptimal);

  std::array<vk::SubpassDescription2, 3> subpasses;

  subpasses[0] = vk::SubpassDescription2(
    vk::SubpassDescriptionFlags(),
    vk::PipelineBindPoint::eGraphics,
    0,
    0,
    nullptr,
    1,
    &colorAttachmentRef,
    &colorResolveAttachmentRef,
    &depthAttachmentRef
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
    nullptr
  );
  subpasses[2] = vk::SubpassDescription2(
    vk::SubpassDescriptionFlags(),
    vk::PipelineBindPoint::eGraphics,
    0,
    0,
    nullptr,
    1,
    &colorResolveAttachmentRef
  );

  if (msaaSamples == vk::SampleCountFlagBits::e1)
  {
    colorAttachment.loadOp = vk::AttachmentLoadOp::eDontCare;
    colorResolveAttachment.loadOp = vk::AttachmentLoadOp::eClear;

    subpasses[0].pColorAttachments = &colorResolveAttachmentRef;
    subpasses[0].pResolveAttachments = nullptr;
  }

  std::vector<vk::AttachmentDescription2> attachments
  {
    colorAttachment,
    depthAttachment,
    colorResolveAttachment
  };

  std::array<vk::SubpassDependency2, 2> dependencies;

  dependencies[0] = vk::SubpassDependency2(
    VK_SUBPASS_EXTERNAL,
    0,
    vk::PipelineStageFlagBits::eColorAttachmentOutput,
    vk::PipelineStageFlagBits::eColorAttachmentOutput,
    vk::AccessFlagBits::eNone,
    vk::AccessFlagBits::eNone
  );
  dependencies[1] = vk::SubpassDependency2(
    0,
    2,
    vk::PipelineStageFlagBits::eColorAttachmentOutput,
    vk::PipelineStageFlagBits::eColorAttachmentOutput,
    vk::AccessFlagBits::eNone,
    vk::AccessFlagBits::eNone
  );

  // only use the first subpass and first dependency
  auto opaqueRenderPassCI = vk::RenderPassCreateInfo2(
    vk::RenderPassCreateFlags(),
    attachments.size(),
    &attachments[0],
    1,
    subpasses.data(),
    1,
    dependencies.data()
  );
  opaqueGraphicsRenderPass = device->createRenderPass2Unique(opaqueRenderPassCI);

  auto renderPassCI = vk::RenderPassCreateInfo2(
    vk::RenderPassCreateFlags(),
    attachments.size(),
    &attachments[0],
    subpasses.size(),
    subpasses.data(),
    dependencies.size(),
    dependencies.data()
  );
  graphicsRenderPass = device->createRenderPass2Unique(renderPassCI);
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

  if (type == PIPELINE_OPAQUE) {
    options.emplace_back("OPAQUE");
    return;
  }

  if (GPUindexing) {
    options.emplace_back("GPUINDEXING");
  }
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

  auto viewport = vk::Viewport(
    0.0f,
    0.0f,
    static_cast<float>(backbufferExtent.width),
    static_cast<float>(backbufferExtent.height),
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

  depthStencilCI.depthTestEnable = VK_TRUE;
  depthStencilCI.depthWriteEnable = enableDepthWrite;
  depthStencilCI.depthCompareOp = vk::CompareOp::eLess;
  depthStencilCI.depthBoundsTestEnable = VK_FALSE;
  depthStencilCI.minDepthBounds = 0.f;
  depthStencilCI.maxDepthBounds = 1.f;
  depthStencilCI.stencilTestEnable = VK_FALSE;

  auto renderPass = *graphicsRenderPass;

  if (type == PIPELINE_OPAQUE) {
    renderPass = *opaqueGraphicsRenderPass;
  } else if (type == PIPELINE_COUNT || type == PIPELINE_COMPRESS) {
    renderPass = *countRenderPass;
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
    throw std::runtime_error("failed to create pipeline!");
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
                          vk::PolygonMode::eFill,
                          materialShaderOptions,
                          "vertex",
                          "fragment",
                          0);

  for (auto u = 0u; u < PIPELINE_MAX; u++)
    createGraphicsPipeline<PointVertex>
                          (PipelineType(u), pointPipelines[u], vk::PrimitiveTopology::eTriangleList,
                          drawMode,
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

void AsyVkRender::createComputePipeline(vk::UniquePipelineLayout & layout, vk::UniquePipeline & pipeline,
                                        std::string const & shaderFile)
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
    sizeof(std::uint32_t)
  );

  auto pipelineLayoutCI = vk::PipelineLayoutCreateInfo(
    vk::PipelineLayoutCreateFlags(),
    1,
    &*computeDescriptorSetLayout,
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
    throw std::runtime_error("failed to create compute pipeline!");
  else
    pipeline = std::move(result.value);
}

void AsyVkRender::createComputePipelines()
{
  createComputePipeline(sum1PipelineLayout, sum1Pipeline, "sum1");
  createComputePipeline(sum2PipelineLayout, sum2Pipeline, "sum2");
  createComputePipeline(sum3PipelineLayout, sum3Pipeline, "sum3");
}

void AsyVkRender::createAttachments()
{
  colorImg = createImage(backbufferExtent.width, backbufferExtent.height, msaaSamples, backbufferImageFormat,
              vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment,
              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  createImageView(backbufferImageFormat, vk::ImageAspectFlagBits::eColor, colorImg.getImage(), colorImageView);

  depthImg = createImage(backbufferExtent.width, backbufferExtent.height, msaaSamples, vk::Format::eD32Sfloat,
          vk::ImageUsageFlagBits::eDepthStencilAttachment,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
  );
  createImageView(vk::Format::eD32Sfloat, vk::ImageAspectFlagBits::eDepth, depthImg.getImage(), depthImageView);

  depthResolveImg = createImage(backbufferExtent.width, backbufferExtent.height, vk::SampleCountFlagBits::e1, vk::Format::eD32Sfloat,
          vk::ImageUsageFlagBits::eDepthStencilAttachment,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
  );
  createImageView(vk::Format::eD32Sfloat, vk::ImageAspectFlagBits::eDepth, depthResolveImg.getImage(), depthResolveImageView);
}

void AsyVkRender::updateUniformBuffer(uint32_t currentFrame)
{
  if (!newUniformBuffer)
    return;

  UniformBufferObject ubo{ };

  ubo.projViewMat = projViewMat;
  ubo.viewMat = viewMat;
  ubo.normMat = glm::inverse(viewMat);

  memcpy(frameObjects[currentFrame].uboMappedMemory->getCopyPtr(), &ubo, sizeof(ubo));

  newUniformBuffer = false;
}

void AsyVkRender::updateBuffers()
{
  if (shouldUpdateBuffers) {
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

//    device->waitIdle();
    createMaterialAndLightBuffers();
    writeMaterialAndLightDescriptors();

    copyToBuffer(lightBf.getBuffer(), &lights[0], lights.size() * sizeof(Light));
    copyToBuffer(materialBf.getBuffer(), &materials[0], materials.size() * sizeof(camp::Material));

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

vk::UniquePipeline & AsyVkRender::getPipelineType(std::array<vk::UniquePipeline, PIPELINE_MAX> & pipelines, bool count)
{
  if (count) {
    return pipelines[PIPELINE_COUNT];
  }

  if (Opaque) {
    return pipelines[PIPELINE_OPAQUE];
  }

  return pipelines[PIPELINE_TRANSPARENT];
}

void AsyVkRender::beginFrameCommands(vk::CommandBuffer cmd)
{
  currentCommandBuffer = cmd;
  currentCommandBuffer.begin(vk::CommandBufferBeginInfo());
}

void AsyVkRender::beginCountFrameRender(int imageIndex)
{
  std::array<vk::ClearValue, 2> clearColors;

  clearColors[0].depthStencil.depth = 1.f;
  clearColors[0].depthStencil.stencil = 0;
  clearColors[1].depthStencil.depth = 1.f;
  clearColors[1].depthStencil.stencil = 0;

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

  clearColors[0] = vk::ClearValue(Background);
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
  auto const copy = (remesh || data->partial || !rendered || badBuffer) && !copied && !data->copiedThisFrame;

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

void AsyVkRender::drawPoints(FrameObject & object)
{
  drawBuffer(object.pointVertexBuffer,
             object.pointIndexBuffer,
             &pointData,
             getPipelineType(pointPipelines));
  pointData.clear();
}

void AsyVkRender::drawLines(FrameObject & object)
{
  drawBuffer(object.lineVertexBuffer,
             object.lineIndexBuffer,
             &lineData,
             getPipelineType(linePipelines));
  lineData.clear();
}

void AsyVkRender::drawMaterials(FrameObject & object)
{
  drawBuffer(object.materialVertexBuffer,
             object.materialIndexBuffer,
             &materialData,
             getPipelineType(materialPipelines));
  materialData.clear();
}

void AsyVkRender::drawColors(FrameObject & object)
{
  drawBuffer(object.colorVertexBuffer,
             object.colorIndexBuffer,
             &colorData,
             getPipelineType(colorPipelines));
  colorData.clear();
}

void AsyVkRender::drawTriangles(FrameObject & object)
{
  drawBuffer(object.triangleVertexBuffer,
             object.triangleIndexBuffer,
             &triangleData,
             getPipelineType(trianglePipelines));
  triangleData.clear();
}

void AsyVkRender::drawTransparent(FrameObject & object)
{
  drawBuffer(object.transparentVertexBuffer,
             object.transparentIndexBuffer,
             &transparentData,
             getPipelineType(transparentPipelines));

  transparentData.clear();
}

void AsyVkRender::partialSums(FrameObject & object, bool timing/*=false*/, bool bindDescriptors/*=false*/)
{
  auto const writeBarrier = vk::MemoryBarrier(
    vk::AccessFlagBits::eShaderWrite,
    vk::AccessFlagBits::eShaderRead
  );

  vk::CommandBuffer const cmd=timing ? *object.partialSumsCommandBuffer : currentCommandBuffer;

  // Do not bother binding descriptor sets again if we have already done so while timing
  if (!timing || bindDescriptors) {
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *sum1PipelineLayout, 0, 1, &*computeDescriptorSet, 0, nullptr);
  }

  // run sum1
  // Only wait for fragment shaders if we are not timing
  if (!timing) {
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eFragmentShader,
                        vk::PipelineStageFlagBits::eComputeShader,
                        { },
                        1,
                        &writeBarrier,
                        0,
                        nullptr,
                        0,
                        nullptr);
  }
  cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *sum1Pipeline);
  cmd.dispatch(g, 1, 1);

  // run sum2
  auto const BlockSize=ceilquotient(g,localSize);
  cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                      vk::PipelineStageFlagBits::eComputeShader,
                      { },
                      1,
                      &writeBarrier,
                      0,
                      nullptr,
                      0,
                      nullptr);
  cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *sum2Pipeline);
  cmd.pushConstants(*sum2PipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(std::uint32_t), &BlockSize);
  cmd.dispatch(1, 1, 1);

  // run sum3
  auto const Final=elements-1;
  cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                      vk::PipelineStageFlagBits::eComputeShader,
                      { },
                      1,
                      &writeBarrier,
                      0,
                      nullptr,
                      0,
                      nullptr);
  cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *sum3Pipeline);
  cmd.pushConstants(*sum3PipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(std::uint32_t), &Final);
  cmd.dispatch(g, 1, 1);

  // Only set this event if not timing, to not waste time while timing
  if (!timing) {
    cmd.setEvent(*object.sumFinishedEvent, vk::PipelineStageFlagBits::eComputeShader);
  }
}

void AsyVkRender::resizeBlendShader(std::uint32_t maxDepth) {

  maxSize=maxDepth;
  recreateBlendPipeline=true;
}

void AsyVkRender::resizeFragmentBuffer(FrameObject & object) {

  if (GPUindexing) {
    waitForEvent(*object.sumFinishedEvent);

    static const auto feedbackMappedPtr=make_unique<vma::cxx::MemoryMapperLock>(feedbackBf);

    const auto maxDepth=feedbackMappedPtr->getCopyPtr()[0];
    fragments=feedbackMappedPtr->getCopyPtr()[1];

    if (maxDepth>maxSize) {
      resizeBlendShader(maxDepth);
    }
  }

  if (fragments>maxFragments) {

    maxFragments=11*fragments/10;
    device->waitForFences(1, &*object.inComputeFence, VK_TRUE, std::numeric_limits<std::uint64_t>::max());
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

  if (!GPUcompress) {

    currentCommandBuffer.fillBuffer(countBf.getBuffer(), 0, countBufferSize, 0);
  }

  beginCountFrameRender(imageIndex);
  currentCommandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *graphicsPipelineLayout, 0, 1, &*object.descriptorSet, 0, nullptr);

  if (!interlock) {

    drawBuffer(object.pointVertexBuffer,
               object.pointIndexBuffer,
               &pointData,
               getPipelineType(pointPipelines, true),
               false);
    drawBuffer(object.lineVertexBuffer,
               object.lineIndexBuffer,
               &lineData,
               getPipelineType(linePipelines, true),
               false);
    drawBuffer(object.materialVertexBuffer,
               object.materialIndexBuffer,
               &materialData,
               getPipelineType(materialPipelines, true),
               false);
    drawBuffer(object.colorVertexBuffer,
               object.colorIndexBuffer,
               &colorData,
               getPipelineType(colorPipelines, true),
               false);
    drawBuffer(object.triangleVertexBuffer,
               object.triangleIndexBuffer,
               &triangleData,
               getPipelineType(trianglePipelines, true),
               false);
  }

  currentCommandBuffer.nextSubpass(vk::SubpassContents::eInline);

  // draw transparent
  drawBuffer(object.transparentVertexBuffer,
             object.transparentIndexBuffer,
             &transparentData,
             getPipelineType(transparentPipelines, true),
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

    auto info = vk::SubmitInfo();

    info.commandBufferCount = 1;
    info.pCommandBuffers = &currentCommandBuffer;

    renderQueue.submit(1, &info, nullptr);

    vk::Result result;

    do
    {
      result = device->getEventStatus(*object.compressionFinishedEvent);
    } while(result != vk::Result::eEventSet);

    elements=GPUindexing ? p[0] : p[0]-1;
    p[0]=1;
  } else {
    endFrameRender();
    endFrameCommands();
    elements=pixels;
    commandsToSubmit.emplace_back(currentCommandBuffer);
  }

  if (elements==0)
    return;

  unsigned int NSUMS=10000;

  if (GPUindexing) {

    beginFrameCommands(*object.computeCommandBuffer);
    g=ceilquotient(elements,groupSize);
    elements=groupSize*g;

    if(settings::verbose > 3) {
      device->resetEvent(*object.startTimedSumsEvent);
      device->resetEvent(*object.timedSumsFinishedEvent);
      // Start recording commands into partialSumsCommandBuffer
      object.partialSumsCommandBuffer->begin(vk::CommandBufferBeginInfo());

      // Wait to execute the compute shaders util we trigger them from CPU
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
      for(unsigned int i=0; i < NSUMS; ++i) {
        partialSums(object, true, i==0);
      }

      // Signal to the CPU the compute shaders have executed
      object.partialSumsCommandBuffer->setEvent(*object.timedSumsFinishedEvent, vk::PipelineStageFlagBits::eComputeShader);
      object.partialSumsCommandBuffer->end();
    }

    partialSums(object);
    endFrameCommands();
    commandsToSubmit.emplace_back(currentCommandBuffer);
  }

  auto info = vk::SubmitInfo();

  info.commandBufferCount = commandsToSubmit.size();
  info.pCommandBuffers = commandsToSubmit.data();

  renderQueue.submit(1, &info, *object.inComputeFence);

  if (GPUindexing && settings::verbose > 3) {
    // Wait until the render queue isn't being used, so we only time
    // our partial sums calculation
    renderQueue.waitIdle();

    auto partialSumsInfo = vk::SubmitInfo();

    partialSumsInfo.commandBufferCount = 1;
    partialSumsInfo.pCommandBuffers = &*object.partialSumsCommandBuffer;

    // Send all the partial sum commands to the GPU, where they will wait
    // until we signal them through the startTimedSums event
    renderQueue.submit(1, &partialSumsInfo, nullptr);

    // Start recording the time
    utils::stopWatch Timer;

    // Signal GPU to start partial sums
    device->setEvent(*object.startTimedSumsEvent);

    // Wait until the GPU tells us the sums are finished
    waitForEvent(*object.timedSumsFinishedEvent);

    // End recording
    double T=Timer.seconds()/NSUMS;
    cout << "elements=" << elements << endl;
    cout << "Tmin (ms)=" << T*1e3 << endl;
    cout << "Megapixels/second=" << elements/T/1e6 << endl;
  }

  if (!GPUindexing) {

    device->waitForFences(1, &*object.inComputeFence, true, std::numeric_limits<std::uint64_t>::max());

    elements=pixels;

    auto offset = offsetStageBfMappedMem->getCopyPtr()+1;
    auto maxsize=countBfMappedMem->getCopyPtr()[0];
    auto count=countBfMappedMem->getCopyPtr()+1;
    size_t Offset=offset[0]=count[0];

    for(size_t i=1; i < elements; ++i)
      offset[i]=Offset += count[i];

    fragments=Offset;

    if (maxsize > maxSize)
      resizeBlendShader(maxsize);

    auto const copy = vk::BufferCopy(
      0, 0, offsetBufferSize
    );

    object.copyCountCommandBuffer->reset();
    object.copyCountCommandBuffer->begin(vk::CommandBufferBeginInfo());
    object.copyCountCommandBuffer->copyBuffer(offsetStageBf.getBuffer(), offsetBf.getBuffer(), 1, &copy);
    object.copyCountCommandBuffer->end();

    auto info = vk::SubmitInfo();

    info.commandBufferCount = 1;
    info.pCommandBuffers = &*object.copyCountCommandBuffer;
    info.signalSemaphoreCount = 1;
    info.pSignalSemaphores = &*object.inCountBufferCopy;

    transferQueue.submit(1, &info, nullptr);
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
  Opaque=transparentData.indices.empty();
  auto transparent=!Opaque;

  if (transparent) {

    device->waitForFences(1, &*object.inComputeFence, VK_TRUE, std::numeric_limits<uint64_t>::max());
    device->resetFences(1, &*object.inComputeFence);
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
  auto transparent=!Opaque;
  beginFrameCommands(getFrameCommandBuffer());

  if (transparent && !GPUindexing) {
    currentCommandBuffer.fillBuffer(countBf.getBuffer(), 0, countBufferSize, 0);
  }

  beginGraphicsFrameRender(imageIndex);
  currentCommandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *graphicsPipelineLayout, 0, 1, &*object.descriptorSet, 0, nullptr);
  drawPoints(object);
  drawLines(object);
  drawMaterials(object);
  drawColors(object);
  drawTriangles(object);


  if (transparent) {

    currentCommandBuffer.nextSubpass(vk::SubpassContents::eInline);
    drawTransparent(object);
    currentCommandBuffer.nextSubpass(vk::SubpassContents::eInline);
    blendFrame(imageIndex);
  }

  endFrameRender();
  endFrameCommands();
}

void AsyVkRender::drawFrame()
{
#ifdef HAVE_PTHREAD
  static bool first=true;
  if(vkthread && first) {
    wait(initSignal,initLock);
    endwait(initSignal,initLock);
    first=false;
  }
#endif

  auto& frameObject = frameObjects[currentFrame];

  // check to see if any pipeline state changed.
  if (recreatePipeline)
  {
//    device->waitIdle();
    createGraphicsPipelines();
    recreatePipeline = false;
  }

  uint32_t imageIndex=0; // index of the current swap chain image to render to

  // Get the image index from the swapchain if not viewing.
  if (View) {
    auto const result = device->acquireNextImageKHR(*swapChain, std::numeric_limits<uint64_t>::max(),
                                                        *frameObject.imageAvailableSemaphore, nullptr,
                                                        &imageIndex);
    if (result == vk::Result::eErrorOutOfDateKHR
        || result == vk::Result::eSuboptimalKHR)
      return recreateSwapChain();
    else if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR)
      throw std::runtime_error("Failed to acquire next swapchain image.");
  }

  device->waitForFences(1, &*frameObject.inFlightFence, VK_TRUE, std::numeric_limits<uint64_t>::max());
  device->resetFences(1, &*frameObject.inFlightFence);
  frameObject.commandBuffer->reset(vk::CommandBufferResetFlagBits());

  updateUniformBuffer(currentFrame);
  updateBuffers();
  resetFrameCopyData();
  preDrawBuffers(frameObject, imageIndex);
  drawBuffers(frameObject, imageIndex);

  std::vector<vk::Semaphore> waitSemaphores {}, signalSemaphores {};

  // Wait for the swapchain to get an image if we are rendering onscreen
  // Also signal to the swapchain that the render has finished if rendering onscreen
  if (View) {
    waitSemaphores.emplace_back(*frameObject.imageAvailableSemaphore);
    signalSemaphores.emplace_back(*frameObject.renderFinishedSemaphore);
  }

  std::vector<vk::PipelineStageFlags> waitStages {vk::PipelineStageFlagBits::eColorAttachmentOutput};

  if (!GPUindexing && !Opaque) {
    waitSemaphores.emplace_back(*frameObject.inCountBufferCopy);
    waitStages.emplace_back(vk::PipelineStageFlagBits::eFragmentShader);
  }

  auto submitInfo = vk::SubmitInfo(
    waitSemaphores.size(),
    waitSemaphores.data(),
    waitStages.data(),
    1,
    &*frameObject.commandBuffer,
    VEC_VIEW(signalSemaphores)
  );

  if (renderQueue.submit(1, &submitInfo, *frameObject.inFlightFence) != vk::Result::eSuccess)
    throw std::runtime_error("failed to submit draw command buffer!");

  // Present to the swapchain if we are rendering on-screen.
  if (View) {
    try
    {
      auto presentInfo = vk::PresentInfoKHR(VEC_VIEW(signalSemaphores), 1, &*swapChain, &imageIndex);
      auto const result = presentQueue.presentKHR(presentInfo);
      if (result == vk::Result::eErrorOutOfDateKHR
          || result == vk::Result::eSuboptimalKHR
          || framebufferResized)
        framebufferResized = false, recreateSwapChain();
      else if (result != vk::Result::eSuccess)
        throw std::runtime_error( "Failed to present swapchain image." );
    }
    catch(std::exception const & e)
    {
      if (std::string(e.what()).find("ErrorOutOfDateKHR") != std::string::npos)
        framebufferResized = false, recreateSwapChain();
      else
      {
        std::cout << "Other error: " << e.what() << std::endl;
        throw;
      }
    }
  }

  if (recreateBlendPipeline) {
    device->waitForFences(1, &*frameObject.inFlightFence, VK_TRUE, std::numeric_limits<uint64_t>::max());
    createBlendPipeline();
    recreateBlendPipeline=false;
  }

  if(queueExport) {
    device->waitForFences(1, &*frameObject.inFlightFence, VK_TRUE, std::numeric_limits<uint64_t>::max());
    Export(imageIndex);
    queueExport=false;
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
    timespec req;
    timespec rem;
    req.tv_sec=(unsigned int) delay;
    req.tv_nsec=(unsigned int) (1.0e9*(delay-req.tv_sec));
    while(nanosleep(&req,&rem) < 0 && errno == EINTR)
      req=rem;
  }
  if(Step) Animate=false;
}

void AsyVkRender::display()
{
  setProjection();

  if(remesh) {
    clearCenters();

    for (int i = 0; i < maxFramesInFlight; i++) {
      frameObjects[i].reset();
    }
  }

  double perspective = orthographic ? 0.0 : 1.0 / Zmax;
  double diagonalSize = hypot(width, height);

  pic->render(diagonalSize, triple(xmin, ymin, Zmin), triple(xmax, ymax, Zmax), perspective, remesh);

  drawFrame();

  if (mode != DRAWMODE_OUTLINE)
    remesh = false;

  static auto const fps = settings::verbose > 2;
  static auto framecount = 0;
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
    if(Oldpid != 0 && waitpid(Oldpid,NULL,WNOHANG) != Oldpid) {
      kill(Oldpid,SIGHUP);
      Oldpid=0;
    }
  }
}

void AsyVkRender::poll()
{
  if (View) {
    vkexit |= glfwWindowShouldClose(window);
  }

  if (vkexit) {
    exitHandler(0);
    vkexit=false;
  }

  if (View) {
    glfwPollEvents();
  }
}

void AsyVkRender::mainLoop()
{
  int nFrames = 0;

  while (poll(), true) {

    if (redraw || queueExport) {
      redraw = false;
      display();
    }

    if (currentIdleFunc != nullptr)
      currentIdleFunc();

    if (!View && nFrames > maxFramesInFlight)
      break;

    nFrames++;
  }

//  vkDeviceWaitIdle(*device);

  if(!View) {
    if(vkthread) {
      if(havewindow) {
        readyAfterExport=true;
#ifdef HAVE_PTHREAD
        pthread_kill(mainthread,SIGUSR1);
#endif
      } else {
        initialized=true;
        readyAfterExport=true;
        Signal(SIGUSR1,exportHandler);
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
  projViewMat = glm::mat4(projMat * viewMat);
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

  pointData.partial=false;
  lineData.partial=false;
  materialData.partial=false;
  colorData.partial=false;
  triangleData.partial=false;
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
        sumUp += T4ij*R1;
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

  return projection(orthographic,vCamera,vUp,vTarget,Zoom0,
                    2.0*atan(tan(0.5*Angle)/Zoom0)/radians,
                    pair(X/width+Shift.getx(),
                         Y/height+Shift.gety()));
}

void AsyVkRender::exportHandler(int) {

  vk->readyAfterExport=true;
  vk->Export(0);
}

void AsyVkRender::Export(int imageIndex) {

  exportCommandBuffer->reset();
  device->resetFences(1, &*exportFence);
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

  VmaAllocationCreateInfo allocInfo = {};
  allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
  allocInfo.memoryTypeBits = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  vk::BufferCreateInfo bufCreateInfo(
          {},
          size,
          vk::BufferUsageFlagBits::eTransferDst);

  vma::cxx::UniqueBuffer exportBuf = allocator.createBuffer(bufCreateInfo, allocInfo);
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
    0,
    nullptr,
    nullptr,
    1,
    &*exportCommandBuffer,
    0,
    nullptr
  );

  if (renderQueue.submit(1, &submitInfo, *exportFence) != vk::Result::eSuccess)
    throw std::runtime_error("failed to submit draw command buffer!");

  device->waitForFences(1, &*exportFence, VK_TRUE, std::numeric_limits<uint64_t>::max());

  vma::cxx::MemoryMapperLock mappedMemory(exportBuf);

  auto * fmt = new unsigned char[backbufferExtent.width * backbufferExtent.height * 3]; // 3 for RGB

  for (auto i = 0u; i < backbufferExtent.height; i++)
    for (auto j = 0u; j < backbufferExtent.width; j++)
      for (auto k = 0u; k < 3; k++)
        // need to flip vertically and swap byte order due to little endian in image data
        // 4 for sizeof unsigned (RGBA)
        fmt[(backbufferExtent.height-1-i)*backbufferExtent.width*3+j*3+(2-k)]=mappedMemory.getCopyPtr<unsigned char>()[i*backbufferExtent.width*4+j*4+k];

  picture pic;
  double w=oWidth;
  double h=oHeight;
  double Aspect=((double) backbufferExtent.width)/backbufferExtent.height;
  if(w > h*Aspect) w=(int) (h*Aspect+0.5);
  else h=(int) (w/Aspect+0.5);

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
  remesh=true;
  setProjection();

#ifdef HAVE_VULKAN
  redraw=true;
#endif

#ifdef HAVE_PTHREAD
  if(vkthread && readyAfterExport) {
    readyAfterExport=false;
    endwait(readySignal,readyLock);
  }
#endif
  exporting=false;
}

void AsyVkRender::quit()
{
#ifdef HAVE_VULKAN
  if(vkthread) {
    bool animating=settings::getSetting<bool>("animating");
    if(animating)
      settings::Setting("interrupt")=true;
    travelHome();
    Animate=settings::getSetting<bool>("autoplay");
#ifdef HAVE_PTHREAD
    if(!interact::interactive || animating) {
      idle();
      endwait(readySignal,readyLock);
    }
#endif
    if(interact::interactive)
      glfwHideWindow(window);
  } else {
    glfwDestroyWindow(window);
    window = nullptr;
    exit(0);
  }
#endif
}

#ifdef HAVE_VULKAN

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
  glm::dmat4 tmpRot(1.0);
  tmpRot=glm::rotate(tmpRot,glm::radians(step),glm::dvec3(1,0,0));
  rotateMat=tmpRot*rotateMat;

  update();
}

void AsyVkRender::rotateY(double step)
{
  glm::dmat4 tmpRot(1.0);
  tmpRot=glm::rotate(tmpRot,glm::radians(step),glm::dvec3(0,1,0));
  rotateMat=tmpRot*rotateMat;

  update();
}

void AsyVkRender::rotateZ(double step)
{
  glm::dmat4 tmpRot(1.0);
  tmpRot=glm::rotate(tmpRot,glm::radians(step),glm::dvec3(0,0,1));
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
    cout << "," << endl << indent << "viewportshift=" << P.viewportshift*Zoom0;
  if(!orthographic)
    cout << "," << endl << indent << "autoadjust=false";
  cout << ");" << endl;
}

void AsyVkRender::shift(double dx, double dy)
{
  double Zoominv=1.0/Zoom0;

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
    cy += dy * (ymax - ymin) / height;
    update();
  }
}

void AsyVkRender::capzoom()
{
  static double maxzoom=sqrt(DBL_MAX);
  static double minzoom=1.0/maxzoom;
  if(Zoom0 <= minzoom) Zoom0=minzoom;
  if(Zoom0 >= maxzoom) Zoom0=maxzoom;

  if(Zoom0 != lastZoom) remesh=true;
  lastZoom=Zoom0;
}

void AsyVkRender::zoom(double dx, double dy)
{
  double zoomFactor=settings::getSetting<double>("zoomfactor");

  if (zoomFactor == 0.0)
    return;

  double zoomStep=settings::getSetting<double>("zoomstep");
  const double limit=log(0.1*DBL_MAX)/log(zoomFactor);
  double stepPower=zoomStep*dy;
  if(fabs(stepPower) < limit) {
    Zoom0 *= std::pow(zoomFactor,-stepPower);
    capzoom();
    update();
  }
}

bool AsyVkRender::capsize(int& w, int& h) {

  bool resize=false;
  if(width > screenWidth) {
    width=screenWidth;
    resize=true;
  }
  if(height > screenHeight) {
    height=screenHeight;
    resize=true;
  }
  return resize;
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
    } else {
      glfwGetWindowPos(window, &x, &y);
      glfwSetWindowPos(window, max(x-2,0), max(y-2, 0));
    }
  }

  reshape0(w,h);
  update();
}

void AsyVkRender::fullscreen(bool reposition) {

  width=screenWidth;
  height=screenHeight;

  if (firstFit) {

    if (width < height*Aspect) {
      Zoom0 *= width/(height*Aspect);
    }
    capzoom();
    setProjection();
    firstFit=false;
  }
  Xfactor=((double) screenHeight)/height;
  Yfactor=((double) screenWidth)/width;
  reshape0(width, height);
  glfwSetWindowSize(window, width, height);
  if (reposition) {
    glfwSetWindowPos(window, 0, 0);
  }
}

void AsyVkRender::reshape0(int Width, int Height) {

  X=(X/Width)*Width;
  Y=(Y/Height)*Height;

  width=Width;
  height=Height;

  static int lastWidth=1;
  static int lastHeight=1;
  if(View && width*height > 1 && (width != lastWidth || height != lastHeight)
     && settings::verbose > 1) {
    cout << "Rendering " << stripDir(Prefix) << " as "
         << width << "x" << height << " image" << endl;
    lastWidth=width;
    lastHeight=height;
  }

  setProjection();
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

  Fitscreen = (Fitscreen + 1) % 3;
  fitscreen();
}

void AsyVkRender::travelHome(bool webgl) {
  X = Y = cx = cy = 0;
  rotateMat = viewMat = glm::mat4(1.0);
  Zoom0 = 1.0;
  update();
}

void AsyVkRender::cycleMode() {
  mode = DrawMode((mode + 1) % DRAWMODE_MAX);
  recreatePipeline = true;
  remesh = true;
  redraw = true;

  if (mode == DRAWMODE_NORMAL) {
    ibl=settings::getSetting<bool>("ibl");
  }
  if (mode == DRAWMODE_OUTLINE) {
    ibl=false;
  }
}

#endif

#endif

} // namespace camp

#endif // HAVE_LIBGLM
