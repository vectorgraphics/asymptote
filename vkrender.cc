#include "vkrender.h"
#include "picture.h"
#include "drawimage.h"

/*
look into subpasses again https://vulkan-tutorial.com/Drawing_a_triangle/Graphics_pipeline_basics/Render_passes

TODO: replace everything with ARR_VIEW

keep validation as a define because the user might not have the vulkan SDK installed
 - Could still make it a runtime option

TODO: make naming like CI/Info consistent
TODO: make struct initialization consistent

How to handle image size / zoom when the window is resized?

"Note that we don't recreate the renderpass here for simplicity. In theory it can be possible for the swap chain image format to change during an applications' lifetime, e.g. when moving a window from an standard range to an high dynamic range monitor. This may require the application to recreate the renderpass to make sure the change between dynamic ranges is properly reflected."

What is the variable 'outlinemode' for?

What about 'home' function?

do this (https://stackoverflow.com/questions/62182124/most-generally-correct-way-of-updating-a-vertex-buffer-in-vulkan) to skip waitForIdle on vertex buffer update?

Tasks for today:
- remove glrender
- add other vulkan pipelines
- add no display mode
- finish up most Vulkan stuff

TODO: put consts everywhere?
*/

void exitHandler(int);

namespace camp
{

std::vector<const char*> instanceExtensions = {
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
        VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME,
#ifdef VALIDATION
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
#endif
};

std::vector<const char*> deviceExtensions = {};

std::vector<const char*> validationLayers = {
#ifdef VALIDATION
        "VK_LAYER_KHRONOS_validation",
#endif
};

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

void AsyVkRender::setDimensions(int width, int height, double x, double y)
{
  double aspect = ((double) width) / height;
  double xshift = (x / (double) width + Shift.getx() * xfactor) * Zoom0;
  double yshift = (y / (double) height + Shift.gety() * yfactor) * Zoom0;
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
  setDimensions(width, height, x, y);

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

void AsyVkRender::initWindow()
{
  double pixelRatio = settings::getSetting<double>("devicepixelratio");

  if (!this->options.display)
    return;

  if (!window) {

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window = glfwCreateWindow(width, height, options.title.data(), nullptr, nullptr);
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
  app->x = (app->x / app->width) * width;
  app->y = (app->y / app->height) * height;
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
  static bool first = true;

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
      //toggleFitScreen();
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
      //idle();
      break;
    case 'M':
      app->cycleMode();
      break;
    case 'E':
      app->queueExport = true;
      break;
    case 'C':
      //showCamera();
      break;
    case '+':
    case '=':
    case '>':
      //expand();
      break;
    case '-':
    case '_':
    case '<':
      //shrink();
      break;
    case 'p':
      // if(getSetting<bool>("reverse")) Animate=false;
      // Setting("reverse")=Step=false;
      // animate();
      break;
    case 'r':
      // if(!getSetting<bool>("reverse")) Animate=false;
      // Setting("reverse")=true;
      // Step=false;
      // animate();
      break;
    case ' ':
      // Step=true;
      // animate();
      break;
    case 17: // Ctrl-q
    case 'Q':
      if(!app->Format.empty()) app->Export(0);
      app->quit();
      break;
  }
}

AsyVkRender::~AsyVkRender()
{
  if (this->options.display) {
    glfwDestroyWindow(this->window);
    glfwTerminate();
  }
}

void AsyVkRender::vkrender(const picture* pic, const string& format,
                           double w, double h, double angle, double zoom,
                           const triple& mins, const triple& maxs, const pair& shift,
                           const pair& margin, double* t,
                           double* background, size_t nlightsin, triple* lights,
                           double* diffuse, double* specular, bool view, int oldpid/*=0*/)
{
  // Do not query disabled devices
  setenv("DRI_PRIME","1",0);

  this->pic = pic;
  this->Format = format;
  this->updateLights = true;
  this->redraw = true;
  this->remesh = true;
  this->nlights = nlightsin;
  this->Lights = lights;
  this->LightsDiffuse = diffuse;
  this->Oldpid = oldpid;

  this->Angle = angle * M_PI / 180.0;
  this->Shift = shift / zoom;
  this->Margin = margin;
  this->View = view;

  Xmin = mins.getx();
  Xmax = maxs.getx();
  Ymin = mins.gety();
  Ymax = maxs.gety();
  Zmin = mins.getz();
  Zmax = maxs.getz();

  orthographic = (this->Angle == 0.0);
  H = orthographic ? 0.0 : -tan(0.5 * this->Angle) * Zmax;
  xfactor = yfactor = 1.0;

  for (int i = 0; i < 4; i++)
    this->Background[i] = static_cast<float>(background[i]);

  this->Zoom0 = zoom;

#ifdef HAVE_PTHREAD
  if(vkthread && vkinit) {
    if(View) {
#ifdef __MSDOS__ // Signals are unreliable in MSWindows
      vkupdate=true;
#else
      pthread_kill(mainthread,SIGUSR1);
#endif
    } //else readyAfterExport=queueExport=true;
    return;
  }
#endif

  if(GPUindexing) {
    localSize=settings::getSetting<Int>("GPUlocalSize");
    blockSize=settings::getSetting<Int>("GPUblockSize");
    groupSize=localSize*blockSize;
  }

  if (vkinit) {
    return;
  }

  clearMaterials();

  rotateMat = glm::mat4(1.0);
  viewMat = glm::mat4(1.0);

  // hardcode this for now
  bool format3d = true;
  double expand = 1.0;

  ArcballFactor = 1 + 8.0 * hypot(Margin.getx(), Margin.gety()) / hypot(w, h);

  antialias=settings::getSetting<Int>("multisample")>1;
  oWidth = w;
  oHeight = h;
  aspect=w/h;

  pair maxtile=settings::getSetting<pair>("maxtile");
  int maxTileWidth=(int) maxtile.getx();
  int maxTileHeight=(int) maxtile.gety();

  if(maxTileWidth <= 0)
    maxTileWidth=1024;
  if(maxTileHeight <= 0)
    maxTileHeight=768;

  int mx, my, workWidth, workHeight;

  glfwInit();
  glfwGetMonitorWorkarea(glfwGetPrimaryMonitor(), &mx, &my, &workWidth, &workHeight);
  screenWidth=workWidth;
  screenHeight=workHeight;

  // Force a hard viewport limit to work around direct rendering bugs.
  // Alternatively, one can use -glOptions=-indirect (with a performance
  // penalty).
  pair maxViewport=settings::getSetting<pair>("maxviewport");
  int maxWidth=maxViewport.getx() > 0 ? (int) ceil(maxViewport.getx()) :
    screenWidth;
  int maxHeight=maxViewport.gety() > 0 ? (int) ceil(maxViewport.gety()) :
    screenHeight;
  if(maxWidth <= 0) maxWidth=max(maxHeight,2);
  if(maxHeight <= 0) maxHeight=max(maxWidth,2);

  if(screenWidth <= 0) screenWidth=maxWidth;
  else screenWidth=min(screenWidth,maxWidth);
  if(screenHeight <= 0) screenHeight=maxHeight;
  else screenHeight=min(screenHeight,maxHeight);

  fullWidth=(int) ceil(expand*w);
  fullHeight=(int) ceil(expand*h);

  if(!format3d) {
    width=fullWidth;
    height=fullHeight;
  } else {
    width=screenWidth;
    height=screenHeight;

    if(width > height*aspect)
      width=min((int) (ceil(height*aspect)),screenWidth);
    else
      height=min((int) (ceil(width/aspect)),screenHeight);
  }

  initWindow();
  initVulkan();
  vkinit=true;
  update();
  mainLoop();
}

void AsyVkRender::initVulkan()
{
  frameObjects.resize(options.maxFramesInFlight);

  createInstance();
  if (options.display) createSurface();
  pickPhysicalDevice();
  createLogicalDevice();
  if (options.display) createSwapChain();
  if (options.display) createImageViews();
  createCommandPools();
  createCommandBuffers();
  createSyncObjects();

  createDescriptorSetLayout();
  createComputeDescriptorSetLayout();

  createBuffers();

  createDescriptorPool();
  createComputeDescriptorPool();
  createDescriptorSets();
  createComputeDescriptorSet();

  createMaterialRenderPass();
  createGraphicsPipelines();
  createComputePipelines();

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

  vkDeviceWaitIdle(*device);

  createSwapChain();
  createImageViews();
  createMaterialRenderPass();
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
  uint32_t glfwExtensionCount = 0;
  const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
  std::set<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
  for (auto& extension : instanceExtensions) extensions.insert(extension);
  return std::vector<const char*>(extensions.begin(), extensions.end());
}

void AsyVkRender::createInstance()
{
  // TODO: replace with asy version
  auto appInfo = vk::ApplicationInfo("Asymptote", VK_MAKE_VERSION(1, 0, 0), "No Engine", VK_MAKE_VERSION(1, 0, 0), VK_API_VERSION_1_0);
  auto extensions = getRequiredInstanceExtensions();
  auto supportedExtensions = getInstanceExtensions();
  if (supportedExtensions.find(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME) != supportedExtensions.end()) {
    extensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
    hasExternalMemoryCapabilitiesExtension = true;
  }
  auto instanceFlags = vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
  auto instanceCI = vk::InstanceCreateInfo(instanceFlags, &appInfo, VEC_VIEW(validationLayers), VEC_VIEW(extensions));
  instance = vk::createInstanceUnique(instanceCI);
}

void AsyVkRender::createSurface()
{
  VkSurfaceKHR surfaceTmp;
  if (glfwCreateWindowSurface(*instance, window, nullptr, &surfaceTmp) != VK_SUCCESS)
    throw std::runtime_error("Failed to create window surface!");
  surface = vk::UniqueSurfaceKHR(surfaceTmp, *instance);
}

void AsyVkRender::pickPhysicalDevice()
{
  auto const getDeviceScore = [this](vk::PhysicalDevice& device) -> std::size_t
  {
    std::size_t score = 0u;

    if (!this->isDeviceSuitable(device))
      return score;

    auto const msaa = getMaxMSAASamples(device);

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

    if (vk::PhysicalDeviceType::eDiscreteGpu == props.deviceType)
      score += 10;
    else if (vk::PhysicalDeviceType::eIntegratedGpu == props.deviceType)
      score += 5;

    return score;
  };

  std::pair<std::size_t, vk::PhysicalDevice> highestDeviceScore { };

  for (auto & dev: instance->enumeratePhysicalDevices())
  {
    auto const score = getDeviceScore(dev);

    if (nullptr == highestDeviceScore.second
        || score > highestDeviceScore.first)
      highestDeviceScore = std::make_pair(score, dev);
  }

  if (0 == highestDeviceScore.first)
    throw std::runtime_error("No suitable GPUs.");

  physicalDevice = highestDeviceScore.second;
  msaaSamples = getMaxMSAASamples(physicalDevice);
}

vk::SampleCountFlagBits AsyVkRender::getMaxMSAASamples( vk::PhysicalDevice & gpu )
{
	vk::PhysicalDeviceProperties props { };

  gpu.getProperties( &props );

	auto const count = props.limits.framebufferColorSampleCounts & props.limits.framebufferDepthSampleCounts;
  auto const maxSamples = settings::getSetting<Int>("multisample");

	if (count & vk::SampleCountFlagBits::e64 && maxSamples >= 64)
		return vk::SampleCountFlagBits::e64;
	if (count & vk::SampleCountFlagBits::e32 && maxSamples >= 32)
		return vk::SampleCountFlagBits::e32;
	if (count & vk::SampleCountFlagBits::e16 && maxSamples >= 16)
		return vk::SampleCountFlagBits::e16;
	if (count & vk::SampleCountFlagBits::e8 && maxSamples >= 8)
		return vk::SampleCountFlagBits::e8;
	if (count & vk::SampleCountFlagBits::e4 && maxSamples >= 4)
		return vk::SampleCountFlagBits::e4;
	if (count & vk::SampleCountFlagBits::e2 && maxSamples >= 2)
		return vk::SampleCountFlagBits::e2;

	return vk::SampleCountFlagBits::e1;
}

// maybe we should prefer using the same queue family for both transfer and render?
// TODO: use if instead of goto and favor same queue family
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
    
    if (VK_FALSE != physicalDevice.getSurfaceSupportKHR(u, *surface)) {
      indices.presentQueueFamily = u,
      indices.presentQueueFamilyFound = true;
    }
    
    if (family.queueFlags & vk::QueueFlagBits::eTransfer) {
      indices.transferQueueFamily = u,
      indices.transferQueueFamilyFound = true;    }
  }

  return indices;
}

bool AsyVkRender::isDeviceSuitable(vk::PhysicalDevice& device)
{
  if (auto const indices = findQueueFamilies(device, options.display ? &*surface : nullptr);
      !indices.transferQueueFamilyFound
      || !indices.renderQueueFamilyFound
      || !(indices.presentQueueFamilyFound || !options.display))
      return false;

  if (!checkDeviceExtensionSupport(device))
    return false;

  if (auto const swapSupport = querySwapChainSupport(device, *surface);
      options.display && (swapSupport.formats.empty() || swapSupport.presentModes.empty()))
    return false;

  auto const features = device.getFeatures();

  return features.samplerAnisotropy;
}

bool AsyVkRender::checkDeviceExtensionSupport(vk::PhysicalDevice& device)
{
  auto extensions = device.enumerateDeviceExtensionProperties();
  std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());
  if (options.display) requiredExtensions.insert(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

  for (auto& extension : extensions) {
    requiredExtensions.erase(extension.extensionName);
  }
  return requiredExtensions.empty();
}

void AsyVkRender::createLogicalDevice()
{
  std::vector<const char*> extensions(deviceExtensions.begin(), deviceExtensions.end());

  std::set<std::string> supportedDeviceExtensions = getDeviceExtensions(physicalDevice);
  if (supportedDeviceExtensions.find("VK_KHR_portability_subset") != supportedDeviceExtensions.end()) {
    extensions.push_back("VK_KHR_portability_subset");
  }
  if (hasExternalMemoryCapabilitiesExtension && supportedDeviceExtensions.find(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME) != supportedDeviceExtensions.end()) {
    extensions.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
    hasExternalMemoryExtension = true;
  }
  if (hasExternalMemoryExtension && supportedDeviceExtensions.find(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME) != supportedDeviceExtensions.end()) {
    extensions.push_back(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
    // this probably won't work because of minImportedHostPointerAlignment and importing the same memory to a device twice can fail
    // hasExternalMemoryHostExtension = true;
  }

  if (options.display) {
    extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  }

  queueFamilyIndices = findQueueFamilies(physicalDevice, options.display ? &*surface : nullptr);

  std::vector<vk::DeviceQueueCreateInfo> queueCIs;
  std::set<uint32_t> uniqueQueueFamilies = {queueFamilyIndices.transferQueueFamily, queueFamilyIndices.renderQueueFamily, queueFamilyIndices.presentQueueFamily};

  float queuePriority = 1.0f;
  for (uint32_t queueFamily : uniqueQueueFamilies) {
    vk::DeviceQueueCreateInfo queueCI(vk::DeviceQueueCreateFlags(), queueFamily, 1, &queuePriority);
    queueCIs.push_back(queueCI);
  }

  vk::PhysicalDeviceFeatures deviceFeatures;

  // for wireframe, alternative draw modes
  deviceFeatures.fillModeNonSolid = true;

  auto deviceCI = vk::DeviceCreateInfo(vk::DeviceCreateFlags(), VEC_VIEW(queueCIs), VEC_VIEW(validationLayers), VEC_VIEW(extensions), &deviceFeatures);

  device = physicalDevice.createDeviceUnique(deviceCI, nullptr);
  transferQueue = device->getQueue(queueFamilyIndices.transferQueueFamily, 0);
  renderQueue = device->getQueue(queueFamilyIndices.renderQueueFamily, 0);
  presentQueue = device->getQueue(queueFamilyIndices.presentQueueFamily, 0);
}

SwapChainSupportDetails AsyVkRender::querySwapChainSupport(vk::PhysicalDevice device, vk::SurfaceKHR& surface)
{
  SwapChainSupportDetails details;

  details.capabilities = device.getSurfaceCapabilitiesKHR(surface);
  details.formats = device.getSurfaceFormatsKHR(surface);
  details.presentModes = device.getSurfacePresentModesKHR(surface);
  details.capabilities.supportedUsageFlags |= vk::ImageUsageFlagBits::eTransferSrc;

  return details;
}

vk::SurfaceFormatKHR AsyVkRender::chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats)
{
  for (const auto& availableFormat : availableFormats) {
    if (availableFormat.format == vk::Format::eB8G8R8A8Uint &&
        availableFormat.colorSpace == vk::ColorSpaceKHR::eAdobergbLinearEXT) {
      return availableFormat;
    }
  }

  return availableFormats[0];
}

vk::PresentModeKHR AsyVkRender::chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes)
{
  for (const auto& availablePresentMode : availablePresentModes) {
    if (availablePresentMode == options.presentMode) {
      return options.presentMode;
    }
  }

  return vk::PresentModeKHR::eFifo;
}

vk::Extent2D AsyVkRender::chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities)
{
  if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
    return capabilities.currentExtent;
  } else {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    vk::Extent2D actualExtent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};

    actualExtent.width = std::min(std::max(actualExtent.width, capabilities.minImageExtent.width),
                                  capabilities.maxImageExtent.width);
    actualExtent.height = std::min(std::max(actualExtent.height, capabilities.minImageExtent.height),
                                   capabilities.maxImageExtent.height);

    return actualExtent;
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
  auto swapChainSupport = querySwapChainSupport(physicalDevice, *surface);

  vk::SurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
  vk::PresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
  vk::Extent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

  uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
  if (swapChainSupport.capabilities.maxImageCount > 0 &&
      imageCount > swapChainSupport.capabilities.maxImageCount) {
    imageCount = swapChainSupport.capabilities.maxImageCount;
  }

  vk::SwapchainCreateInfoKHR swapchainCI = vk::SwapchainCreateInfoKHR(vk::SwapchainCreateFlagsKHR(), *surface, imageCount, surfaceFormat.format, surfaceFormat.colorSpace, extent, 1, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc, vk::SharingMode::eExclusive, 0, nullptr, swapChainSupport.capabilities.currentTransform, vk::CompositeAlphaFlagBitsKHR::eOpaque, presentMode, VK_TRUE, nullptr, nullptr);
  if (*swapChain)
    swapchainCI.oldSwapchain = *swapChain;

  if (queueFamilyIndices.renderQueueFamily != queueFamilyIndices.presentQueueFamily) {
    static uint32_t indices[] = {queueFamilyIndices.renderQueueFamily,queueFamilyIndices.presentQueueFamily};

    swapchainCI.imageSharingMode = vk::SharingMode::eConcurrent;
    swapchainCI.queueFamilyIndexCount = 2;
    swapchainCI.pQueueFamilyIndices= indices;
  }

  swapChain = device->createSwapchainKHRUnique(swapchainCI, nullptr);
  swapChainImages = device->getSwapchainImagesKHR(*swapChain);
  swapChainImageFormat = surfaceFormat.format;
  swapChainExtent = extent;
}

void AsyVkRender::createImageViews()
{
  swapChainImageViews.resize(swapChainImages.size());
  for (size_t i = 0; i < swapChainImages.size(); i++) {
    vk::ImageViewCreateInfo viewCI = vk::ImageViewCreateInfo(vk::ImageViewCreateFlags(), swapChainImages[i], vk::ImageViewType::e2D, swapChainImageFormat, vk::ComponentMapping(), vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
    swapChainImageViews[i] = device->createImageViewUnique(viewCI, nullptr);
  }
}

vk::UniqueShaderModule AsyVkRender::createShaderModule(const std::vector<char>& code)
{
  //glslang::InitializeProcess();

  auto shaderModuleCI =
          vk::ShaderModuleCreateInfo(vk::ShaderModuleCreateFlags(), code.size(),
                                     reinterpret_cast<const uint32_t*>(code.data()));
  return device->createShaderModuleUnique(shaderModuleCI);
}

// how will this work with multiple pipelines and without a swapchain?
void AsyVkRender::createFramebuffers()
{
  swapChainFramebuffers.resize(swapChainImageViews.size());
  for (auto i = 0u; i < swapChainImageViews.size(); i++)
  {
    vk::ImageView attachments[] = {*colorImageView, *depthImageView, *swapChainImageViews[i]};
    auto framebufferCI = vk::FramebufferCreateInfo(
      vk::FramebufferCreateFlags(),
      *materialRenderPass,
      ARR_VIEW(attachments),
      swapChainExtent.width,
      swapChainExtent.height,
      1
    );
    swapChainFramebuffers[i] = device->createFramebufferUnique(framebufferCI);
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
  auto allocInfo = vk::CommandBufferAllocateInfo(*renderCommandPool, vk::CommandBufferLevel::ePrimary, static_cast<uint32_t>(options.maxFramesInFlight));
  auto commandBuffers = device->allocateCommandBuffersUnique(allocInfo);
  for (int i = 0; i < options.maxFramesInFlight; i++)
    frameObjects[i].commandBuffer = std::move(commandBuffers[i]);
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
  cmd.end();

  auto info = vk::SubmitInfo();

  info.commandBufferCount = 1;
  info.pCommandBuffers = &cmd;

  renderQueue.submit(1, &info, nullptr);
  renderQueue.waitIdle();

  device->freeCommandBuffers(*renderCommandPool, 1, &cmd);
}

void AsyVkRender::createSyncObjects()
{
  for (size_t i = 0; i < options.maxFramesInFlight; i++) {
    frameObjects[i].imageAvailableSemaphore = device->createSemaphoreUnique(vk::SemaphoreCreateInfo());
    frameObjects[i].renderFinishedSemaphore = device->createSemaphoreUnique(vk::SemaphoreCreateInfo());
    frameObjects[i].inFlightFence = device->createFenceUnique(vk::FenceCreateInfo(vk::FenceCreateFlagBits::eSignaled));
  }
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

void AsyVkRender::createBuffer(vk::Buffer& buffer, vk::DeviceMemory& bufferMemory, vk::BufferUsageFlags usage,
                               vk::MemoryPropertyFlags properties, vk::DeviceSize size)
{
  auto bufferCI = vk::BufferCreateInfo(vk::BufferCreateFlags(), size, usage, vk::SharingMode::eExclusive);
  buffer = device->createBuffer(bufferCI);

  auto memRequirements = device->getBufferMemoryRequirements(buffer);
  uint32_t memoryTypeIndex = selectMemory(memRequirements, properties);
  auto memoryCI = vk::MemoryAllocateInfo(memRequirements.size, memoryTypeIndex);
  bufferMemory = device->allocateMemory(memoryCI);
  device->bindBufferMemory(buffer, bufferMemory, 0);
}

// TODO: try without unique? (then use for staging buffers)
void AsyVkRender::createBufferUnique(vk::UniqueBuffer& buffer, vk::UniqueDeviceMemory& bufferMemory,
                                     vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties,
                                     vk::DeviceSize size)
{
  // TODO: sharing mode and queue family indices
  auto bufferCI = vk::BufferCreateInfo(vk::BufferCreateFlags(), size, usage);
  buffer = device->createBufferUnique(bufferCI);

  auto memRequirements = device->getBufferMemoryRequirements(*buffer);
  uint32_t memoryTypeIndex = selectMemory(memRequirements, properties);
  auto memoryAI = vk::MemoryAllocateInfo(memRequirements.size, memoryTypeIndex);
  bufferMemory = device->allocateMemoryUnique(memoryAI);
  device->bindBufferMemory(*buffer, *bufferMemory, 0);
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

void AsyVkRender::copyToBuffer(const vk::Buffer& buffer, const void* data, vk::DeviceSize size,
                               vk::Buffer stagingBuffer, vk::DeviceMemory stagingBufferMemory)
{
  if (hasExternalMemoryHostExtension) {
    auto externalMemoryBufferCI = vk::ExternalMemoryBufferCreateInfo(vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT);
    auto bufferCI = vk::BufferCreateInfo(vk::BufferCreateFlags(), size, vk::BufferUsageFlagBits::eTransferSrc, vk::SharingMode::eExclusive, 0, nullptr, &externalMemoryBufferCI);
    auto hostBuffer = device->createBufferUnique(bufferCI);
    // ERROR: How to bind this?
    copyBufferToBuffer(*hostBuffer, buffer, size);
  } else {
    bool cleanupStagingBuffer = false;
    if (stagingBuffer || stagingBufferMemory) {
      if (!(stagingBuffer && stagingBufferMemory))
        throw std::runtime_error("staging buffer and memory must be both set or both null!");
    } else {
      createBuffer(stagingBuffer, stagingBufferMemory, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, size);
      cleanupStagingBuffer = true;
    }

    void* memoryPtr = device->mapMemory(stagingBufferMemory, 0, size, vk::MemoryMapFlags());
    memcpy(memoryPtr, data, size);
    device->unmapMemory(stagingBufferMemory);

    copyBufferToBuffer(stagingBuffer, buffer, size);

    if (cleanupStagingBuffer) {
      device->destroyBuffer(stagingBuffer);
      device->freeMemory(stagingBufferMemory);
    }
  }
}

void AsyVkRender::createImage(std::uint32_t w, std::uint32_t h, vk::SampleCountFlagBits samples, vk::Format fmt,
                              vk::ImageUsageFlags usage, vk::MemoryPropertyFlags props, vk::UniqueImage & img,
                              vk::UniqueDeviceMemory & mem)
{
  auto info = vk::ImageCreateInfo();

  info.imageType      = vk::ImageType::e2D;
  info.extent         = vk::Extent3D(w, h, 1);
  info.mipLevels      = 1;
  info.arrayLayers    = 1;
  info.format         = fmt;
  info.tiling         = vk::ImageTiling::eOptimal;
  info.initialLayout  = vk::ImageLayout::eUndefined;
  info.usage          = usage;
  info.sharingMode    = vk::SharingMode::eExclusive;
  info.samples        = samples;

  img = device->createImageUnique(info);

  auto const req = device->getImageMemoryRequirements(*img);

  vk::MemoryAllocateInfo alloc(
    req.size,
    selectMemory(req, props)
  );

  mem = device->allocateMemoryUnique(alloc);
  device->bindImageMemory(*img, *mem, 0);
}

void AsyVkRender::createImageView(vk::Format fmt, vk::ImageAspectFlagBits flags,
                                  vk::UniqueImage& img, vk::UniqueImageView& imgView)
{
  auto info = vk::ImageViewCreateInfo();

  info.image = *img;
  info.viewType = vk::ImageViewType::e2D;
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
  vk::UniqueBuffer stagingBuffer;
  vk::UniqueDeviceMemory stagingBufferMemory;

  createBufferUnique(stagingBuffer, stagingBufferMemory, vk::BufferUsageFlagBits::eTransferDst,
                     vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, size);
  
  auto const cmd = beginSingleCommands();
  auto const cpy = vk::BufferCopy(
    0, 0, size
  );

  cmd.copyBuffer(buffer, *stagingBuffer, 1, &cpy);

  endSingleCommands(cmd);

  void* memoryPtr = device->mapMemory(*stagingBufferMemory, 0, size, vk::MemoryMapFlags());
  memcpy(data, memoryPtr, size);
  device->unmapMemory(*stagingBufferMemory);
}

void AsyVkRender::setDeviceBufferData(DeviceBuffer& buffer, const void* data, vk::DeviceSize size, std::size_t nobjects)
{
  // Vulkan doesn't allow a buffer to have a size of 0
  auto bufferCI = vk::BufferCreateInfo(vk::BufferCreateFlags(), std::max(vk::DeviceSize(1), size), buffer.usage);
  buffer.buffer = device->createBufferUnique(bufferCI);

  auto memRequirements = device->getBufferMemoryRequirements(*buffer.buffer);
  uint32_t memoryTypeIndex = selectMemory(memRequirements, buffer.properties);
  buffer.nobjects = nobjects;
  if (size > buffer.memorySize || buffer.memorySize == 0) {
    // minimum array size of 16 bytes to avoid some Vulkan issues
    auto newSize = 16;
    while (newSize < size) newSize *= 2;
    buffer.memorySize = newSize;
    auto memoryAI = vk::MemoryAllocateInfo(buffer.memorySize, memoryTypeIndex);
    buffer.memory = device->allocateMemoryUnique(memoryAI);

    // check whether we need a staging buffer
    if (!hasExternalMemoryHostExtension) {
      createBufferUnique(buffer.stagingBuffer, buffer.stagingBufferMemory, vk::BufferUsageFlagBits::eTransferSrc,
                         vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer.memorySize);
    }
  } else {
    // TODO: downsize memory?
  }

  device->bindBufferMemory(*buffer.buffer, *buffer.memory, 0);
  if (data) {
    if (hasExternalMemoryHostExtension) {
      copyToBuffer(*buffer.buffer, data, size);
    } else {
      copyToBuffer(*buffer.buffer, data, size, *buffer.stagingBuffer, *buffer.stagingBufferMemory);
    }

    buffer.memorySize = size;
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

  std::vector<vk::DescriptorSetLayoutBinding> layoutBindings {
    uboLayoutBinding,
    materialBufferBinding,
    lightBufferBinding,
    countBufferBinding
  };

  auto layoutCI = vk::DescriptorSetLayoutCreateInfo(
    vk::DescriptorSetLayoutCreateFlags(),
    layoutBindings.size(),
    &layoutBindings[0]
  );
  materialDescriptorSetLayout = device->createDescriptorSetLayoutUnique(layoutCI);
}

void AsyVkRender::createComputeDescriptorSetLayout()
{
  std::vector< vk::DescriptorSetLayoutBinding > layoutBindings
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
  std::array<vk::DescriptorPoolSize, 4> poolSizes;

  poolSizes[0].type = vk::DescriptorType::eUniformBuffer;
  poolSizes[0].descriptorCount = options.maxFramesInFlight;

  poolSizes[1].type = vk::DescriptorType::eStorageBuffer;
  poolSizes[1].descriptorCount = options.maxFramesInFlight;

  poolSizes[2].type = vk::DescriptorType::eStorageBuffer;
  poolSizes[2].descriptorCount = options.maxFramesInFlight;

  poolSizes[3].type = vk::DescriptorType::eStorageBuffer;
  poolSizes[3].descriptorCount = options.maxFramesInFlight;

  auto poolCI = vk::DescriptorPoolCreateInfo(
    vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
    options.maxFramesInFlight,
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
  std::vector<vk::DescriptorSetLayout> layouts(options.maxFramesInFlight, *materialDescriptorSetLayout);
  auto allocInfo = vk::DescriptorSetAllocateInfo(*descriptorPool, VEC_VIEW(layouts));
  auto descriptorSets = device->allocateDescriptorSetsUnique(allocInfo);

  for (size_t i = 0; i < options.maxFramesInFlight; i++) {
    frameObjects[i].descriptorSet = std::move(descriptorSets[i]);

    auto uboInfo = vk::DescriptorBufferInfo();

    uboInfo.buffer = *frameObjects[i].uniformBuffer;
    uboInfo.offset = 0;
    uboInfo.range = sizeof(UniformBufferObject);

    auto materialBufferInfo = vk::DescriptorBufferInfo();

    materialBufferInfo.buffer = *materialBuffer;
    materialBufferInfo.offset = 0;
    materialBufferInfo.range = sizeof(camp::Material) * NMaterials;

    auto lightBufferInfo = vk::DescriptorBufferInfo();

    lightBufferInfo.buffer = *lightBuffer;
    lightBufferInfo.offset = 0;
    lightBufferInfo.range = sizeof(Light) * nlights;

    auto countBufferInfo = vk::DescriptorBufferInfo();

    countBufferInfo.buffer = *countBuffer;
    countBufferInfo.offset = 0;
    countBufferInfo.range = countBufferSize;

    std::array<vk::WriteDescriptorSet, 4> writes;

    writes[0].dstSet = *frameObjects[i].descriptorSet;
    writes[0].dstBinding = 0;
    writes[0].dstArrayElement = 0;
    writes[0].descriptorType = vk::DescriptorType::eUniformBuffer;
    writes[0].descriptorCount = 1;
    writes[0].pBufferInfo = &uboInfo;

    writes[1].dstSet = *frameObjects[i].descriptorSet;
    writes[1].dstBinding = 1;
    writes[1].dstArrayElement = 0;
    writes[1].descriptorType = vk::DescriptorType::eStorageBuffer;
    writes[1].descriptorCount = 1;
    writes[1].pBufferInfo = &materialBufferInfo;

    writes[2].dstSet = *frameObjects[i].descriptorSet;
    writes[2].dstBinding = 2;
    writes[2].dstArrayElement = 0;
    writes[2].descriptorType = vk::DescriptorType::eStorageBuffer;
    writes[2].descriptorCount = 1;
    writes[2].pBufferInfo = &lightBufferInfo;

    writes[3].dstSet = *frameObjects[i].descriptorSet;
    writes[3].dstBinding = 3;
    writes[3].dstArrayElement = 0;
    writes[3].descriptorType = vk::DescriptorType::eStorageBuffer;
    writes[3].descriptorCount = 1;
    writes[3].pBufferInfo = &countBufferInfo;

    device->updateDescriptorSets(writes.size(), &writes[0], 0, nullptr);
  }
}

void AsyVkRender::createComputeDescriptorSet()
{
  auto allocInfo = vk::DescriptorSetAllocateInfo(
    *computeDescriptorPool,
    1,
    &*computeDescriptorSetLayout
  );

  computeDescriptorSet = std::move(device->allocateDescriptorSetsUnique(allocInfo)[0]);

  auto countBufferInfo = vk::DescriptorBufferInfo();

  countBufferInfo.buffer = *countBuffer;
  countBufferInfo.offset = 0;
  countBufferInfo.range = countBufferSize;

  auto globalSumBufferInfo = vk::DescriptorBufferInfo();

  globalSumBufferInfo.buffer = *globalSumBuffer;
  globalSumBufferInfo.offset = 0;
  globalSumBufferInfo.range = globalSize;

  auto offsetBufferInfo = vk::DescriptorBufferInfo();

  offsetBufferInfo.buffer = *offsetBuffer;
  offsetBufferInfo.offset = 0;
  offsetBufferInfo.range = offsetBufferSize;

  auto feedbackBufferInfo = vk::DescriptorBufferInfo();

  feedbackBufferInfo.buffer = *feedbackBuffer;
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

void AsyVkRender::createBuffers()
{
  pixels=(swapChainExtent.width+1)*(swapChainExtent.height+1);
  GLuint Pixels;

  if (GPUindexing) {
    GLuint G=ceilquotient(pixels,groupSize);
    Pixels=groupSize*G;
    globalSize=localSize*ceilquotient(G,localSize);
  }
  else {
    Pixels=pixels;
  }

  countBufferSize=(Pixels+2)*sizeof(std::uint32_t);
  offsetBufferSize=(Pixels+2)*sizeof(std::uint32_t);
  feedbackBufferSize=2*sizeof(std::uint32_t);

  createBufferUnique(materialBuffer,
                     materialBufferMemory,
                     vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
                     vk::MemoryPropertyFlagBits::eDeviceLocal,
                     sizeof(camp::Material) * NMaterials);

  createBufferUnique(lightBuffer,
                     lightBufferMemory,
                     vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
                     vk::MemoryPropertyFlagBits::eDeviceLocal,
                     sizeof(camp::Light) * nlights);

  createBufferUnique(countBuffer,
                     countBufferMemory,
                     vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
                     vk::MemoryPropertyFlagBits::eDeviceLocal,
                     countBufferSize);

  createBufferUnique(globalSumBuffer,
                     globalSumBufferMemory,
                     vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
                     vk::MemoryPropertyFlagBits::eDeviceLocal,
                     globalSize);

  createBufferUnique(offsetBuffer,
                     offsetBufferMemory,
                     vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
                     vk::MemoryPropertyFlagBits::eDeviceLocal,
                     offsetBufferSize);

  createBufferUnique(feedbackBuffer,
                     feedbackBufferMemory,
                     vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
                     vk::MemoryPropertyFlagBits::eDeviceLocal,
                     feedbackBufferSize);

  uint32_t i = 50;
  copyToBuffer(*countBuffer, &i, 4);

  for (size_t i = 0; i < options.maxFramesInFlight; i++) {

    createBufferUnique(frameObjects[i].uniformBuffer,
                       frameObjects[i].uniformBufferMemory,
                       vk::BufferUsageFlagBits::eUniformBuffer,
                       vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                       sizeof(UniformBufferObject));
    frameObjects[i].uboData = device->mapMemory(*frameObjects[i].uniformBufferMemory, 0, sizeof(UniformBufferObject), vk::MemoryMapFlags());
  }
}

void AsyVkRender::createMaterialRenderPass()
{
  auto colorAttachment = vk::AttachmentDescription(
    vk::AttachmentDescriptionFlags(),
    swapChainImageFormat,
    msaaSamples,
    vk::AttachmentLoadOp::eClear,
    vk::AttachmentStoreOp::eStore,
    vk::AttachmentLoadOp::eDontCare,
    vk::AttachmentStoreOp::eDontCare,
    vk::ImageLayout::eUndefined,
    vk::ImageLayout::eColorAttachmentOptimal
  );
  auto colorResolveAttachment = vk::AttachmentDescription(
    vk::AttachmentDescriptionFlags(),
    swapChainImageFormat,
    vk::SampleCountFlagBits::e1,
    vk::AttachmentLoadOp::eDontCare,
    vk::AttachmentStoreOp::eStore,
    vk::AttachmentLoadOp::eDontCare,
    vk::AttachmentStoreOp::eDontCare,
    vk::ImageLayout::eUndefined,
    vk::ImageLayout::ePresentSrcKHR
  );
  auto depthAttachment = vk::AttachmentDescription(
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

  depthAttachment.initialLayout = vk::ImageLayout::eUndefined;
  depthAttachment.finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

  auto colorAttachmentRef = vk::AttachmentReference(0, vk::ImageLayout::eColorAttachmentOptimal);
  auto depthAttachmentRef = vk::AttachmentReference(1, vk::ImageLayout::eDepthStencilAttachmentOptimal);
  auto colorResolveAttachmentRef = vk::AttachmentReference(2, vk::ImageLayout::eColorAttachmentOptimal);

  auto subpass = vk::SubpassDescription(
    vk::SubpassDescriptionFlags(),
    vk::PipelineBindPoint::eGraphics,
    0,
    nullptr,
    1,
    &colorAttachmentRef
  );

  subpass.pResolveAttachments = &colorResolveAttachmentRef;
  subpass.pDepthStencilAttachment = &depthAttachmentRef;

  if (msaaSamples == vk::SampleCountFlagBits::e1)
  {
    colorAttachment.loadOp = vk::AttachmentLoadOp::eDontCare;
    colorResolveAttachment.loadOp = vk::AttachmentLoadOp::eClear;
    subpass.pColorAttachments = &colorResolveAttachmentRef;
    subpass.pResolveAttachments = nullptr;
  }

  std::vector< vk::AttachmentDescription > attachments {colorAttachment, depthAttachment, colorResolveAttachment};

  auto dependency = vk::SubpassDependency();

  dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
  dependency.dstSubpass = 0;
  dependency.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput
                            | vk::PipelineStageFlagBits::eEarlyFragmentTests;
  dependency.srcAccessMask = vk::AccessFlagBits::eNone;
  dependency.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput
                            | vk::PipelineStageFlagBits::eEarlyFragmentTests;
  dependency.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite
                              | vk::AccessFlagBits::eDepthStencilAttachmentWrite;

  auto renderPassCI = vk::RenderPassCreateInfo(vk::RenderPassCreateFlags(), attachments.size(), &attachments[0], 1, &subpass, 1, &dependency);

  materialRenderPass = device->createRenderPassUnique(renderPassCI, nullptr);
}

template<typename V>
void AsyVkRender::createGraphicsPipeline(vk::UniquePipelineLayout & layout, vk::UniquePipeline & graphicsPipeline,
                                         vk::UniquePipeline & countPipeline, vk::PrimitiveTopology topology,
                                         vk::PolygonMode fillMode, std::string const & shaderFile)
{
  auto vertShaderCode = readFile("shaders/" + shaderFile + ".vert.spv");
  auto fragShaderCode = readFile("shaders/" + shaderFile + ".frag.spv");
  auto countShaderCode = readFile("shaders/count.frag.spv");

  vk::UniqueShaderModule vertShaderModule = createShaderModule(vertShaderCode);
  vk::UniqueShaderModule fragShaderModule = createShaderModule(fragShaderCode);
  vk::UniqueShaderModule countShaderModule = createShaderModule(countShaderCode);

  vk::SpecializationMapEntry specializationMapEntries[] = {};
  uint32_t specializationData[] = {};
  auto specializationInfo = vk::SpecializationInfo(ARR_VIEW(specializationMapEntries), RAW_VIEW(specializationData));

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
  auto countShaderStageCI = vk::PipelineShaderStageCreateInfo(
    vk::PipelineShaderStageCreateFlags(),
    vk::ShaderStageFlagBits::eFragment,
    *countShaderModule,
    "main",
    &specializationInfo
  );
  vk::PipelineShaderStageCreateInfo defaultStages[] = {vertShaderStageCI, fragShaderStageCI};
  vk::PipelineShaderStageCreateInfo countStages[] = {vertShaderStageCI, countShaderStageCI};

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

  auto viewport = vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f);
  auto scissor = vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent);
  auto viewportStateCI = vk::PipelineViewportStateCreateInfo(vk::PipelineViewportStateCreateFlags(), 1, &viewport, 1, &scissor);

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
    msaaSamples,
    VK_FALSE,
    0.0f,
    nullptr,
    VK_FALSE,
    VK_FALSE
  );

  auto colorBlendAttachment = vk::PipelineColorBlendAttachmentState(VK_FALSE, vk::BlendFactor::eZero, vk::BlendFactor::eZero, vk::BlendOp::eAdd, vk::BlendFactor::eZero, vk::BlendFactor::eZero, vk::BlendOp::eAdd, vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);

  auto colorBlendCI = vk::PipelineColorBlendStateCreateInfo(vk::PipelineColorBlendStateCreateFlags(), VK_FALSE, vk::LogicOp::eCopy, 1, &colorBlendAttachment, {0.0f, 0.0f, 0.0f, 0.0f});

  auto depthStencilCI = vk::PipelineDepthStencilStateCreateInfo();

  depthStencilCI.depthTestEnable = VK_TRUE;
  depthStencilCI.depthWriteEnable = VK_TRUE;
  depthStencilCI.depthCompareOp = vk::CompareOp::eLess;
  depthStencilCI.depthBoundsTestEnable = VK_FALSE;
  depthStencilCI.minDepthBounds = 0.f;
  depthStencilCI.maxDepthBounds = 1.f;
  depthStencilCI.stencilTestEnable = VK_FALSE;

  auto flagsPushConstant = vk::PushConstantRange(
    vk::ShaderStageFlagBits::eFragment,
    0,
    sizeof(PushConstants)
  );

  auto pipelineLayoutCI = vk::PipelineLayoutCreateInfo(vk::PipelineLayoutCreateFlags(), 1, &*materialDescriptorSetLayout, 0, nullptr);

  pipelineLayoutCI.pPushConstantRanges = &flagsPushConstant;
  pipelineLayoutCI.pushConstantRangeCount = 1;

  layout = device->createPipelineLayoutUnique(pipelineLayoutCI, nullptr);

  auto const makePipeline = [=](vk::UniquePipeline & pipeline,
                                vk::PipelineShaderStageCreateInfo * stages)
  {
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
      *materialPipelineLayout,
      *materialRenderPass,
      0,
      nullptr
    );

    if (auto result = device->createGraphicsPipelineUnique(nullptr, pipelineCI, nullptr);
        result.result != vk::Result::eSuccess)
      throw std::runtime_error("failed to create pipeline!");
    else
      pipeline = std::move(result.value);
  };

  makePipeline(graphicsPipeline, defaultStages);
  makePipeline(countPipeline, countStages);
}

void AsyVkRender::createGraphicsPipelines()
{
  createGraphicsPipeline<MaterialVertex>
                         (materialPipelineLayout, materialPipeline,
                         materialCountPipeline,
                         vk::PrimitiveTopology::eTriangleList,
                         (options.mode == DRAWMODE_WIREFRAME) ? vk::PolygonMode::eLine : vk::PolygonMode::eFill,
                         "material");
  createGraphicsPipeline<ColorVertex>
                         (colorPipelineLayout, colorPipeline,
                         colorCountPipeline,
                         vk::PrimitiveTopology::eTriangleList,
                         (options.mode == DRAWMODE_WIREFRAME) ? vk::PolygonMode::eLine : vk::PolygonMode::eFill,
                         "color");
  createGraphicsPipeline<ColorVertex>
                         (trianglePipelineLayout, trianglePipeline,
                         triangleCountPipeline,
                         vk::PrimitiveTopology::eTriangleList,
                         (options.mode == DRAWMODE_WIREFRAME) ? vk::PolygonMode::eLine : vk::PolygonMode::eFill,
                         "triangle");
  createGraphicsPipeline<MaterialVertex>
                         (linePipelineLayout, linePipeline,
                         lineCountPipeline,
                         vk::PrimitiveTopology::eLineList,
                         vk::PolygonMode::eFill,
                         "material");
  createGraphicsPipeline<PointVertex>
                         (pointPipelineLayout, pointPipeline,
                         pointCountPipeline,
                         vk::PrimitiveTopology::ePointList,
                         vk::PolygonMode::ePoint,
                         "point");
}

void AsyVkRender::createComputePipeline(vk::UniquePipelineLayout & layout, vk::UniquePipeline & pipeline,
                                        std::string const & shaderFile)
{
  auto computeShaderCode = readFile("shaders/" + shaderFile + ".comp.spv");

  vk::UniqueShaderModule computeShaderModule = createShaderModule(computeShaderCode);

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

  if (auto result = device->createComputePipelineUnique(VK_NULL_HANDLE, computePipelineCI);
      result.result != vk::Result::eSuccess)
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
  createImage(swapChainExtent.width, swapChainExtent.height, msaaSamples, swapChainImageFormat,
              vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment,
              vk::MemoryPropertyFlagBits::eDeviceLocal, colorImage, colorImageMemory);
  createImageView(swapChainImageFormat, vk::ImageAspectFlagBits::eColor, colorImage, colorImageView);

  createImage(swapChainExtent.width, swapChainExtent.height, msaaSamples, vk::Format::eD32Sfloat,
              vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::MemoryPropertyFlagBits::eDeviceLocal, depthImage,
              depthImageMemory);
  createImageView(vk::Format::eD32Sfloat, vk::ImageAspectFlagBits::eDepth, depthImage, depthImageView);
}

void AsyVkRender::updateUniformBuffer(uint32_t currentFrame)
{
  if (!newUniformBuffer)
    return;

  UniformBufferObject ubo{ };

  ubo.projViewMat = projViewMat;
  ubo.viewMat = viewMat;
  ubo.normMat = glm::inverse(viewMat);

  memcpy(frameObjects[currentFrame].uboData, &ubo, sizeof(ubo));

  newUniformBuffer = false;
}

void AsyVkRender::updateBuffers()
{
  if (updateLights) {
    std::vector<Light> lights;

    for (int i = 0; i < nlights; i++)
      lights.emplace_back(
        Light {
          {Lights[i].getx(), Lights[i].gety(), Lights[i].getz(), 0.f},
          {static_cast<float>(LightsDiffuse[4 * i]),
          static_cast<float>(LightsDiffuse[4 * i + 1]),
          static_cast<float>(LightsDiffuse[4 * i + 2]), 0.f}
        }
      );

    copyToBuffer(*lightBuffer, &lights[0], lights.size() * sizeof(Light));
    updateLights=false;
  }

  if (materials != oldMaterials) {

    copyToBuffer(*materialBuffer, &materials[0], materials.size() * sizeof(camp::Material));
    oldMaterials = materials;
  }

}

PushConstants AsyVkRender::buildPushConstants()
{
  auto pushConstants = PushConstants { };

  pushConstants.constants[0] = options.mode!= DRAWMODE_NORMAL ? 0 : nlights;
  pushConstants.constants[1] = swapChainExtent.width;
  
  return pushConstants;
}

vk::CommandBuffer & AsyVkRender::getFrameCommandBuffer()
{
  return *frameObjects[currentFrame].commandBuffer;
}

void AsyVkRender::beginFrameCommands(vk::CommandBuffer cmd)
{
  currentCommandBuffer = cmd;
  currentCommandBuffer.begin(vk::CommandBufferBeginInfo());
}

void AsyVkRender::beginFrameRender(vk::Framebuffer framebuffer)
{
  std::array<vk::ClearValue, 3> clearColors;

  clearColors[0] = vk::ClearValue(Background);
  clearColors[1].depthStencil.depth = 1.f;
  clearColors[1].depthStencil.stencil = 0;
  clearColors[2] = vk::ClearValue(Background);

  auto renderPassInfo = vk::RenderPassBeginInfo(
    *materialRenderPass,
    framebuffer,
    vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent),
    clearColors.size(),
    &clearColors[0]
  );

  currentCommandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
}

void AsyVkRender::beginFrame(vk::Framebuffer framebuffer, vk::CommandBuffer cmd)
{
  materialData.copiedThisFrame=false;
  colorData.copiedThisFrame=false;
  triangleData.copiedThisFrame=false;
  transparentData.copiedThisFrame=false;
  lineData.copiedThisFrame=false;
  pointData.copiedThisFrame=false;

  beginFrameCommands(cmd);
  beginFrameRender(framebuffer);
}

void AsyVkRender::recordCommandBuffer(DeviceBuffer & vertexBuffer,
                                      DeviceBuffer & indexBuffer,
                                      VertexBuffer * data,
                                      vk::UniquePipeline & pipeline,
                                      vk::UniquePipelineLayout & pipelineLayout) {
  
  if (data->indices.empty())
    return;

  auto const badBuffer = static_cast<void*>(*vertexBuffer.buffer) == nullptr;
  auto const rendered = data->renderCount >= options.maxFramesInFlight;
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

  std::vector<vk::Buffer> vertexBuffers = {*vertexBuffer.buffer};
  std::vector<vk::DeviceSize> vertexOffsets = {0};
  auto const pushConstants = buildPushConstants();

  currentCommandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline);
  currentCommandBuffer.bindVertexBuffers(0, vertexBuffers, vertexOffsets);
  currentCommandBuffer.bindIndexBuffer(*indexBuffer.buffer, 0, vk::IndexType::eUint32);
  currentCommandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipelineLayout, 0, 1, &*frameObjects[currentFrame].descriptorSet, 0, nullptr);
  currentCommandBuffer.pushConstants(*pipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(PushConstants), &pushConstants);
  currentCommandBuffer.drawIndexed(indexBuffer.nobjects, 1, 0, 0, 0);

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
  recordCommandBuffer(object.pointVertexBuffer,
                      object.pointIndexBuffer,
                      &pointData,
                      pointPipeline,
                      pointPipelineLayout);
  pointData.clear();
}

void AsyVkRender::drawLines(FrameObject & object)
{
  recordCommandBuffer(object.lineVertexBuffer,
                      object.lineIndexBuffer,
                      &lineData,
                      linePipeline,
                      linePipelineLayout);
  lineData.clear();
}

void AsyVkRender::drawMaterials(FrameObject & object)
{
  recordCommandBuffer(object.materialVertexBuffer,
                      object.materialIndexBuffer,
                      &materialData,
                      materialPipeline,
                      materialPipelineLayout);
  materialData.clear();
}

void AsyVkRender::drawColors(FrameObject & object)
{
  recordCommandBuffer(object.colorVertexBuffer,
                      object.colorIndexBuffer,
                      &colorData,
                      colorPipeline,
                      colorPipelineLayout);
  colorData.clear();
}

void AsyVkRender::drawTriangles(FrameObject & object)
{
  recordCommandBuffer(object.triangleVertexBuffer,
                      object.triangleIndexBuffer,
                      &triangleData,
                      trianglePipeline,
                      trianglePipelineLayout);
  triangleData.clear();
}

int ceilquotient(int x, int y)
{
  return (x+y-1)/y;
}

void AsyVkRender::partialSums(bool readSize)
{
  currentCommandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *sum1PipelineLayout, 0, 1, &*computeDescriptorSet, 0, nullptr);

  // run sum1
  currentCommandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *sum1Pipeline);
  currentCommandBuffer.dispatch(g, 1, 1);

  // run sum2
  auto const writeBarrier = vk::MemoryBarrier( // todo sum2 fast
    vk::AccessFlagBits::eShaderRead,
    vk::AccessFlagBits::eShaderWrite
  );
  currentCommandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *sum2Pipeline);
  currentCommandBuffer.pushConstants(*sum2PipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(std::uint32_t), &blockSize);
  currentCommandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, 
                                       vk::PipelineStageFlagBits::eComputeShader,
                                       { },
                                       1,
                                       &writeBarrier, 
                                       0,
                                       nullptr,
                                       0,
                                       nullptr);
  currentCommandBuffer.dispatch(1, 1, 1);

  // run sum3
  auto const Final=elements-1;
  currentCommandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *sum3Pipeline);
  currentCommandBuffer.pushConstants(*sum3PipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(std::uint32_t), &Final);
  currentCommandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, 
                                       vk::PipelineStageFlagBits::eComputeShader,
                                       { },
                                       1,
                                       &writeBarrier, 
                                       0,
                                       nullptr,
                                       0,
                                       nullptr);
  currentCommandBuffer.dispatch(g, 1, 1);
}

void AsyVkRender::resizeBlendShader(std::uint32_t maxDepth) {

}

void AsyVkRender::resizeFragmentBuffer() {

  if (GPUindexing) {
    auto const barrier = vk::BufferMemoryBarrier(
      vk::AccessFlagBits::eShaderRead,
      vk::AccessFlagBits::eShaderWrite,
      queueFamilyIndices.renderQueueFamily,
      queueFamilyIndices.renderQueueFamily,
      *feedbackBuffer,
      0,
      feedbackBufferSize
    );
    currentCommandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                         vk::PipelineStageFlagBits::eFragmentShader,
                                         { },
                                         0,
                                         nullptr,
                                         1,
                                         &barrier,
                                         0,
                                         nullptr);
    std::array<std::uint32_t, 2> feedbackBufferMap;
    copyFromBuffer(*feedbackBuffer, feedbackBufferMap.data(), feedbackBufferSize);
    auto const maxDepth=feedbackBufferMap[0];

    if (maxDepth>maxSize) {
      resizeBlendShader(maxDepth);
    }

    fragments=feedbackBufferMap[1];
  }

  if (fragments>maxFragments) {

    maxFragments=11*fragments/10;
    /// ...
  }
}

void AsyVkRender::refreshBuffers(FrameObject & object, int imageIndex)
{
  if (GPUindexing && !GPUcompress) {
    zeroBuffer(*countBuffer, countBufferSize);
  }

  if (!interlock) {
    recordCommandBuffer(object.pointVertexBuffer,
                        object.pointIndexBuffer,
                        &pointData,
                        pointCountPipeline,
                        pointPipelineLayout);
    recordCommandBuffer(object.lineVertexBuffer,
                        object.lineIndexBuffer,
                        &lineData,
                        lineCountPipeline,
                        linePipelineLayout);
    recordCommandBuffer(object.materialVertexBuffer,
                        object.materialIndexBuffer,
                        &materialData,
                        materialCountPipeline,
                        materialPipelineLayout);
    recordCommandBuffer(object.colorVertexBuffer,
                        object.colorIndexBuffer,
                        &colorData,
                        colorCountPipeline,
                        colorPipelineLayout);
    recordCommandBuffer(object.triangleVertexBuffer,
                        object.triangleIndexBuffer,
                        &triangleData,
                        triangleCountPipeline,
                        trianglePipelineLayout);
  }

  // draw transparent

  endFrameRender();

  if (GPUcompress) {
    // ...
  }
  else {
    elements = pixels;
  }

  if (GPUindexing) {
    
    g=ceilquotient(elements,groupSize);
    elements=groupSize*g;

    if(settings::verbose > 3) {
      static bool first=true;
      if(first) {
        partialSums();
        first=false;
      }
      unsigned int N=10000;
      utils::stopWatch Timer;
      for(unsigned int i=0; i < N; ++i)
        partialSums();
      
      // glFinish(); ??
      double T=Timer.seconds()/N;
      cout << "elements=" << elements << endl;
      cout << "Tmin (ms)=" << T*1e3 << endl;
      cout << "Megapixels/second=" << elements/T/1e6 << endl;
    }

    partialSums(true);
  }
}

void AsyVkRender::drawBuffers(FrameObject & object, int imageIndex)
{
  copied=false;
  Opaque=transparentData.indices.empty();
  auto transparent=!Opaque;

  if (ssbo && transparent || TRUE) { // todo

    refreshBuffers(object, imageIndex);

    if (!interlock) {
      resizeFragmentBuffer();
      copied=true;
    }

    beginFrameRender(*swapChainFramebuffers[imageIndex]);
  }

  drawPoints(object);
  drawLines(object);
  drawMaterials(object);
  drawColors(object);
  drawTriangles(object);

  if (transparent) {
    
    copied=true;

    if (interlock) {
      resizeFragmentBuffer();
    }
  }

  Opaque=0;
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

  if(format3dWait)
    wait(initSignal,initLock);
#endif

  auto& frameObject = frameObjects[currentFrame];

  device->waitForFences(1, &*frameObject.inFlightFence, VK_TRUE, std::numeric_limits<uint64_t>::max());

  // check to see if any pipeline state changed.
  if (recreatePipeline)
  {
    device->waitIdle();
    createGraphicsPipelines();
    recreatePipeline = false;
  }

  uint32_t imageIndex; // index of the current swap chain image to render to
  if (auto const result = device->acquireNextImageKHR(*swapChain, std::numeric_limits<uint64_t>::max(),
                                                      *frameObject.imageAvailableSemaphore, nullptr,
                                                      &imageIndex);
      result == vk::Result::eErrorOutOfDateKHR
      || result == vk::Result::eSuboptimalKHR)
    return recreateSwapChain();
  else if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR)
    throw std::runtime_error("Failed to acquire next swapchain image.");

  device->resetFences(1, &*frameObject.inFlightFence);
  frameObject.commandBuffer->reset(vk::CommandBufferResetFlagBits());
  updateUniformBuffer(currentFrame);
  updateBuffers();

  beginFrame(*swapChainFramebuffers[imageIndex], getFrameCommandBuffer());

  drawBuffers(frameObject, imageIndex);

  endFrame(imageIndex);

  vk::Semaphore waitSemaphores[] = {*frameObject.imageAvailableSemaphore};
  vk::PipelineStageFlags waitStages = vk::PipelineStageFlagBits::eColorAttachmentOutput;
  vk::Semaphore signalSemaphores[] = {*frameObject.renderFinishedSemaphore};
  auto submitInfo = vk::SubmitInfo(ARR_VIEW(waitSemaphores), &waitStages, 1, &*frameObject.commandBuffer, ARR_VIEW(signalSemaphores));

  if (renderQueue.submit(1, &submitInfo, *frameObject.inFlightFence) != vk::Result::eSuccess)
    throw std::runtime_error("failed to submit draw command buffer!");

  {
    {
      // renderQueue.waitIdle();
      // std::uint32_t * data = (std::uint32_t*)(new char[offsetBufferSize]);

      // copyFromBuffer(*offsetBuffer, data, offsetBufferSize);

      // std::cout << "PIXEL0: " << data[0] << std::endl;
      // std::cout << "PIXEL1: " << data[1] << std::endl;
      // std::cout << "PIXEL2: " << data[2] << std::endl;
      // std::cout << "PIXEL3: " << data[3] << std::endl;
      // std::cout << "PIXEL4: " << data[4] << std::endl;
      // std::cout << "PIXEL4: " << data[5] << std::endl;
      // std::cout << "PIXEL4: " << data[6] << std::endl;
      // std::cout << "PIXEL4: " << data[7] << std::endl;
      // std::cout << "PIXEL4: " << data[8] << std::endl;
      // std::cout << "PIXEL4: " << data[9] << std::endl;
      // std::cout << "PIXEL4: " << data[10] << std::endl;
      // std::cout << "PIXEL4: " << data[11] << std::endl;

      // delete[] data;
    }
  }

  auto presentInfo = vk::PresentInfoKHR(ARR_VIEW(signalSemaphores), 1, &*swapChain, &imageIndex);

  try
  {
    if (auto const result = presentQueue.presentKHR(presentInfo);
        result == vk::Result::eErrorOutOfDateKHR
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
      throw;
  }

  if (queueExport)
    Export(imageIndex);

  currentFrame = (currentFrame + 1) % options.maxFramesInFlight;
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
    
    for (int i = 0; i < options.maxFramesInFlight; i++) {
      frameObjects[i].reset();
    }
  }

  double perspective = orthographic ? 0.0 : 1.0 / Zmax;
  double diagonalSize = hypot(width, height);

  pic->render(diagonalSize, triple(xmin, ymin, Zmin), triple(xmax, ymax, Zmax), perspective, remesh);

  drawFrame();

  if (options.mode != DRAWMODE_OUTLINE)
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
if(queueExport) {
    //Export();
    queueExport=false;
  }
  if(!vkthread) {
    if(Oldpid != 0 && waitpid(Oldpid,NULL,WNOHANG) != Oldpid) {
      kill(Oldpid,SIGHUP);
      Oldpid=0;
    }
  }
}

void AsyVkRender::poll() {

  vkexit |= glfwWindowShouldClose(window);

  if (vkexit) {
    exitHandler(0);
    vkexit=false;
  }
  glfwPollEvents();
}

void AsyVkRender::mainLoop()
{
  while (poll(), true) {
    
    if (redraw || queueExport) {
      redraw = false;
      display();
    }

    if (currentIdleFunc != nullptr)
      currentIdleFunc();
  }

  vkDeviceWaitIdle(*device);
}

void AsyVkRender::updateProjection()
{
  projViewMat = glm::mat4(projMat * viewMat);
}

void AsyVkRender::frustum(GLdouble left, GLdouble right, GLdouble bottom,
                          GLdouble top, GLdouble nearVal, GLdouble farVal)
{
  projMat = glm::frustum(left, right, bottom, top, nearVal, farVal);
  updateProjection();
}

void AsyVkRender::ortho(GLdouble left, GLdouble right, GLdouble bottom,
                        GLdouble top, GLdouble nearVal, GLdouble farVal)
{
  projMat = glm::ortho(left, right, bottom, top, nearVal, farVal);
  updateProjection();
}

void AsyVkRender::clearCenters()
{
  camp::drawElement::centers.clear();
  camp::drawElement::centermap.clear();
}

void AsyVkRender::clearMaterials()
{
  materials.clear();
  materials.reserve(nmaterials);
  materialMap.clear();

  pointData.partial=false;
  lineData.partial=false;
  materialData.partial=false;
  colorData.partial=false;
  triangleData.partial=false;
}

void AsyVkRender::Export(int imageIndex) {

  exportCommandBuffer->reset();
  device->resetFences(1, &*exportFence);
  exportCommandBuffer->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

  auto const size = device->getImageMemoryRequirements(swapChainImages[0]).size;
  auto const swapExtent = vk::Extent3D(
    swapChainExtent.width,
    swapChainExtent.height,
    1
  );
  auto const reg = vk::BufferImageCopy(
    0,
    swapChainExtent.width,
    swapChainExtent.height,
    vk::ImageSubresourceLayers(
      vk::ImageAspectFlagBits::eColor, 0, 0, 1
    ),
    { },
    swapExtent
  );
  vk::DeviceMemory mem;
  vk::Buffer dst;
  createBuffer(dst, mem, vk::BufferUsageFlagBits::eTransferDst,
               vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
               size);
  transitionImageLayout(
    *exportCommandBuffer,
    swapChainImages[imageIndex],
    vk::AccessFlagBits::eMemoryRead,
    vk::AccessFlagBits::eTransferRead,
    vk::ImageLayout::ePresentSrcKHR,
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

  exportCommandBuffer->copyImageToBuffer(swapChainImages[imageIndex], vk::ImageLayout::eTransferSrcOptimal, dst, 1, &reg);
  
  transitionImageLayout(
    *exportCommandBuffer,
    swapChainImages[imageIndex],
    vk::AccessFlagBits::eTransferRead,
    vk::AccessFlagBits::eMemoryRead,
    vk::ImageLayout::eTransferSrcOptimal,
    vk::ImageLayout::ePresentSrcKHR,
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
  
  auto * data = static_cast<unsigned char*>(device->mapMemory(mem, 0, size));
  auto * fmt = new unsigned char[swapChainExtent.width * swapChainExtent.height * 3]; // 3 for RGB

  for (int i = 0; i < swapChainExtent.height; i++)
    for (int j = 0; j < swapChainExtent.width; j++)
      for (int k = 0; k < 3; k++)
        // need to flip vertically and swap byte order due to little endian in image data
        // 4 for sizeof unsigned (RGBA)
        fmt[(swapChainExtent.height-1-i)*swapChainExtent.width*3+j*3+(2-k)]=data[i*swapChainExtent.width*4+j*4+k];

  picture pic;
  double w=oWidth;
  double h=oHeight;
  double Aspect=((double) swapChainExtent.width)/swapChainExtent.height;
  if(w > h*Aspect) w=(int) (h*Aspect+0.5);
  else h=(int) (w/Aspect+0.5);

  auto * const Image=new camp::drawRawImage(fmt,
                                            swapChainExtent.width,
                                            swapChainExtent.height,
                                            transform(0.0,0.0,w,0.0,0.0,h),
                                            antialias);
  pic.append(Image);
  pic.shipout(NULL,Prefix,Format,false,View);
  delete Image;
  delete[] fmt;

  device->unmapMemory(mem);

  device->freeMemory(mem);
  device->destroyBuffer(dst);

  queueExport = false;
  remesh=true;
  setProjection();

#ifndef HAVE_LIBOSMESA
#ifdef HAVE_LIBGLUT
  redraw=true;
#endif

#ifdef HAVE_PTHREAD
  if(vkthread && readyAfterExport) {
    readyAfterExport=false;
    endwait(readySignal,readyLock);
  }
#endif
#endif
  exporting=false;
  // camp::initSSBO=true;
}

void AsyVkRender::quit()
{
#ifdef HAVE_LIBOSMESA
  if(osmesa_buffer) delete[] osmesa_buffer;
  if(ctx) OSMesaDestroyContext(ctx);
  exit(0);
#endif
#ifdef HAVE_LIBGLUT
  if(vkthread) {
    bool animating=settings::getSetting<bool>("animating");
    if(animating)
      settings::Setting("interrupt")=true;
    travelHome();
    Animate=settings::getSetting<bool>("autoplay");
#ifdef HAVE_PTHREAD
    if(!interact::interactive || animating) {
      idle();
      //options.display=false;
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

void AsyVkRender::shift(double dx, double dy)
{
  double Zoominv=1.0/Zoom0;

  x += dx*Zoominv;
  y += -dy*Zoominv;
  update();
}

void AsyVkRender::pan(double dx, double dy)
{
  cx += dx * (xmax - xmin) / width;
  cy += dy * (ymax - ymin) / height;
  update();
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

void AsyVkRender::travelHome()
{
  x = y = cx = cy = 0;
  rotateMat = viewMat = glm::mat4(1.0);
  Zoom0 = 1.0;
  update();
}

void AsyVkRender::cycleMode()
{
  options.mode = DrawMode((options.mode + 1) % DRAWMODE_MAX);
  recreatePipeline = true;
  remesh = true;
  redraw = true;
}

} // namespace camp
