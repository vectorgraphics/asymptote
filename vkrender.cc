#include "vkrender.h"
#include "picture.h"
#include "drawimage.h"

#define SHADER_DIRECTORY "base/shaders/"

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

std::vector<const char*> deviceExtensions = {
  "VK_EXT_fragment_shader_interlock",
  "VK_KHR_depth_stencil_resolve",
  "VK_KHR_create_renderpass2",
  "VK_KHR_multiview",
  "VK_KHR_maintenance2"
};

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

  glslang::FinalizeProcess();
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

#if !defined(HAVE_LIBOSMESA)
  GPUindexing=settings::getSetting<bool>("GPUindexing");
  GPUcompress=settings::getSetting<bool>("GPUcompress");
#else
  GPUindexing=false;
  GPUcompress=false;
#endif

  if(GPUindexing) {
    localSize=settings::getSetting<Int>("GPUlocalSize");
    blockSize=settings::getSetting<Int>("GPUblockSize");
    groupSize=localSize*blockSize;
  }

#ifdef HAVE_LIBOSMESA
  interlock=false;
#else
  interlock=settings::getSetting<bool>("GPUinterlock");
#endif

  if (vkinit) {
    return;
  }

  clearMaterials();

  rotateMat = glm::mat4(1.0);
  viewMat = glm::mat4(1.0);

  // hardcode this for now
  bool v3d=format == "v3d";
  bool webgl=format == "html";
  bool format3d=webgl || v3d;

  ArcballFactor = 1 + 8.0 * hypot(Margin.getx(), Margin.gety()) / hypot(w, h);

  antialias=settings::getSetting<Int>("multisample")>1;
  oWidth = w;
  oHeight = h;
  aspect=w/h;

  double expand;
  if(format3d)
    expand=1.0;
  else {
    expand=settings::getSetting<double>("render");
    if(expand < 0)
      expand *= (Format.empty() || Format == "eps" || Format == "pdf")                 ? -2.0 : -1.0;
    if(antialias) expand *= 2.0;
  }

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

  if(format3d) {
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

  travelHome(format3d);
  setProjection();
  if(format3d) {
    remesh=true;
    return;
  }

  setosize();

  Animate=settings::getSetting<bool>("autoplay") && vkthread;

  initWindow();

  if(View) {
    if(!settings::getSetting<bool>("fitscreen"))
      Fitscreen=0;
    firstFit=true;
    fitscreen();
    setosize();
  }

  initVulkan();
  vkinit=true;
  update();
  mainLoop();
}

void AsyVkRender::initVulkan()
{
  if (!glslang::InitializeProcess())
    throw std::runtime_error("Unable to initialize glslang.");

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
  writeDescriptorSets();

  createCountRenderPass();
  createGraphicsRenderPass();
  createGraphicsPipelineLayout();
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

  device->waitIdle();

  createSwapChain();
  createDependentBuffers();
  writeDescriptorSets();
  createImageViews();
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
  uint32_t glfwExtensionCount = 0;
  const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
  std::set<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
  for (auto& extension : instanceExtensions) extensions.insert(extension);
  return std::vector<const char*>(extensions.begin(), extensions.end());
}

void AsyVkRender::createInstance()
{
  // TODO: replace with asy version
  auto appInfo = vk::ApplicationInfo("Asymptote", VK_MAKE_VERSION(1, 0, 0), "No Engine", VK_MAKE_VERSION(1, 0, 0), VK_API_VERSION_1_2);
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
  std::uint32_t nSamples;

  std::tie(nSamples, msaaSamples) = getMaxMSAASamples(physicalDevice);

  if(settings::verbose > 1 && msaaSamples != vk::SampleCountFlagBits::e1)
      cout << "Multisampling enabled with sample width " << nSamples
           << endl;
}

std::pair<std::uint32_t, vk::SampleCountFlagBits> AsyVkRender::getMaxMSAASamples( vk::PhysicalDevice & gpu )
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

  if (interlock) {

    if (supportedDeviceExtensions.find("VK_EXT_fragment_shader_interlock") == supportedDeviceExtensions.end())
      interlock=false;
    else
      extensions.emplace_back("VK_EXT_fragment_shader_interlock");
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

  physicalDevice.getProperties2(&props);

  // for wireframe, alternative draw modes
  deviceFeatures.fillModeNonSolid = true;

  auto deviceCI = vk::DeviceCreateInfo(
    vk::DeviceCreateFlags(),
    VEC_VIEW(queueCIs),
    VEC_VIEW(validationLayers),
    VEC_VIEW(extensions),
    &deviceFeatures,
    &interlockFeatures
  );

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

  vk::SwapchainCreateInfoKHR swapchainCI = vk::SwapchainCreateInfoKHR(
    vk::SwapchainCreateFlagsKHR(),
    *surface,
    imageCount,
    surfaceFormat.format,
    surfaceFormat.colorSpace,
    extent,
    1,
    vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc,
    vk::SharingMode::eExclusive,
    0,
    nullptr,
    swapChainSupport.capabilities.currentTransform,
    vk::CompositeAlphaFlagBitsKHR::eOpaque,
    presentMode,
    VK_TRUE,
    nullptr,
    nullptr
  );

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

TBuiltInResource AsyVkRender::getDefaultShaderResources() {

  TBuiltInResource res;

  res.maxLights = 32;
	res.maxClipPlanes = 6;
	res.maxTextureUnits = 32;
	res.maxTextureCoords = 32;
	res.maxVertexAttribs = 64;
	res.maxVertexUniformComponents = 4096;
	res.maxVaryingFloats = 64;
	res.maxVertexTextureImageUnits = 32;
	res.maxCombinedTextureImageUnits = 80;
	res.maxTextureImageUnits = 32;
	res.maxFragmentUniformComponents = 4096;
	res.maxDrawBuffers = 32;
	res.maxVertexUniformVectors = 128;
	res.maxVaryingVectors = 8;
	res.maxFragmentUniformVectors = 16;
	res.maxVertexOutputVectors = 16;
	res.maxFragmentInputVectors = 15;
	res.minProgramTexelOffset = -8;
	res.maxProgramTexelOffset = 7;
	res.maxClipDistances = 8;
	res.maxComputeWorkGroupCountX = 65535;
	res.maxComputeWorkGroupCountY = 65535;
	res.maxComputeWorkGroupCountZ = 65535;
	res.maxComputeWorkGroupSizeX = 1024;
	res.maxComputeWorkGroupSizeY = 1024;
	res.maxComputeWorkGroupSizeZ = 64;
	res.maxComputeUniformComponents = 1024;
	res.maxComputeTextureImageUnits = 16;
	res.maxComputeImageUniforms = 8;
	res.maxComputeAtomicCounters = 8;
	res.maxComputeAtomicCounterBuffers = 1;
	res.maxVaryingComponents = 60;
	res.maxVertexOutputComponents = 64;
	res.maxGeometryInputComponents = 64;
	res.maxGeometryOutputComponents = 128;
	res.maxFragmentInputComponents = 128;
	res.maxImageUnits = 8;
	res.maxCombinedImageUnitsAndFragmentOutputs = 8;
	res.maxCombinedShaderOutputResources = 8;
	res.maxImageSamples = 0;
	res.maxVertexImageUniforms = 0;
	res.maxTessControlImageUniforms = 0;
	res.maxTessEvaluationImageUniforms = 0;
	res.maxGeometryImageUniforms = 0;
	res.maxFragmentImageUniforms = 8;
	res.maxCombinedImageUniforms = 8;
	res.maxGeometryTextureImageUnits = 16;
	res.maxGeometryOutputVertices = 256;
	res.maxGeometryTotalOutputComponents = 1024;
	res.maxGeometryUniformComponents = 1024;
	res.maxGeometryVaryingComponents = 64;
	res.maxTessControlInputComponents = 128;
	res.maxTessControlOutputComponents = 128;
	res.maxTessControlTextureImageUnits = 16;
	res.maxTessControlUniformComponents = 1024;
	res.maxTessControlTotalOutputComponents = 4096;
	res.maxTessEvaluationInputComponents = 128;
	res.maxTessEvaluationOutputComponents = 128;
	res.maxTessEvaluationTextureImageUnits = 16;
	res.maxTessEvaluationUniformComponents = 1024;
	res.maxTessPatchComponents = 120;
	res.maxPatchVertices = 32;
	res.maxTessGenLevel = 64;
	res.maxViewports = 16;
	res.maxVertexAtomicCounters = 0;
	res.maxTessControlAtomicCounters = 0;
	res.maxTessEvaluationAtomicCounters = 0;
	res.maxGeometryAtomicCounters = 0;
	res.maxFragmentAtomicCounters = 8;
	res.maxCombinedAtomicCounters = 8;
	res.maxAtomicCounterBindings = 1;
	res.maxVertexAtomicCounterBuffers = 0;
	res.maxTessControlAtomicCounterBuffers = 0;
	res.maxTessEvaluationAtomicCounterBuffers = 0;
	res.maxGeometryAtomicCounterBuffers = 0;
	res.maxFragmentAtomicCounterBuffers = 1;
	res.maxCombinedAtomicCounterBuffers = 1;
	res.maxAtomicCounterBufferSize = 16384;
	res.maxTransformFeedbackBuffers = 4;
	res.maxTransformFeedbackInterleavedComponents = 64;
	res.maxCullDistances = 8;
	res.maxCombinedClipAndCullDistances = 8;
	res.maxSamples = 64;
	res.maxMeshOutputVerticesNV = 256;
	res.maxMeshOutputPrimitivesNV = 512;
	res.maxMeshWorkGroupSizeX_NV = 32;
	res.maxMeshWorkGroupSizeY_NV = 1;
	res.maxMeshWorkGroupSizeZ_NV = 1;
	res.maxTaskWorkGroupSizeX_NV = 32;
	res.maxTaskWorkGroupSizeY_NV = 1;
	res.maxTaskWorkGroupSizeZ_NV = 1;
	res.maxMeshViewCountNV = 4;
	res.limits.nonInductiveForLoops = 1;
	res.limits.whileLoops = 1;
	res.limits.doWhileLoops = 1;
	res.limits.generalUniformIndexing = 1;
	res.limits.generalAttributeMatrixVectorIndexing = 1;
	res.limits.generalVaryingIndexing = 1;
	res.limits.generalSamplerIndexing = 1;
	res.limits.generalVariableIndexing = 1;
	res.limits.generalConstantMatrixVectorIndexing = 1;

  return res;
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
  auto const res = getDefaultShaderResources();
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
  depthFramebuffers.resize(swapChainImageViews.size());
  graphicsFramebuffers.resize(swapChainImageViews.size());

  for (auto i = 0u; i < swapChainImageViews.size(); i++)
  {
    vk::ImageView attachments[] = {*colorImageView, *depthImageView, *depthResolveImageView, *swapChainImageViews[i]};
    std::array<vk::ImageView, 1> depthAttachments
    {
      *depthImageView,
      //*depthResolveImageView
    };

    auto depthFramebufferCI = vk::FramebufferCreateInfo(
      {},
      *countRenderPass,
      depthAttachments.size(),
      depthAttachments.data(),
      swapChainExtent.width,
      swapChainExtent.height,
      1
    );
    auto graphicsFramebufferCI = vk::FramebufferCreateInfo(
      vk::FramebufferCreateFlags(),
      *graphicsRenderPass,
      ARR_VIEW(attachments),
      swapChainExtent.width,
      swapChainExtent.height,
      1
    );

    depthFramebuffers[i] = device->createFramebufferUnique(depthFramebufferCI);
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
  auto renderAllocInfo = vk::CommandBufferAllocateInfo(*renderCommandPool, vk::CommandBufferLevel::ePrimary, static_cast<uint32_t>(options.maxFramesInFlight * 4));
  auto transferAllocInfo = vk::CommandBufferAllocateInfo(*transferCommandPool, vk::CommandBufferLevel::ePrimary, static_cast<uint32_t>(options.maxFramesInFlight));
  auto renderCommands = device->allocateCommandBuffersUnique(renderAllocInfo);
  auto transferCommands = device->allocateCommandBuffersUnique(transferAllocInfo);

  for (int i = 0; i < options.maxFramesInFlight; i++)
  {
    frameObjects[i].commandBuffer = std::move(renderCommands[3 * i]);
    frameObjects[i].countCommandBuffer = std::move(renderCommands[3 * i + 1]);
    frameObjects[i].computeCommandBuffer = std::move(renderCommands[3 * i + 2]);
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
  for (size_t i = 0; i < options.maxFramesInFlight; i++) {
    frameObjects[i].imageAvailableSemaphore = device->createSemaphoreUnique(vk::SemaphoreCreateInfo());
    frameObjects[i].renderFinishedSemaphore = device->createSemaphoreUnique(vk::SemaphoreCreateInfo());
    frameObjects[i].inCountBufferCopy = device->createSemaphoreUnique(vk::SemaphoreCreateInfo());
    frameObjects[i].inFlightFence = device->createFenceUnique(vk::FenceCreateInfo(vk::FenceCreateFlagBits::eSignaled));
    frameObjects[i].inComputeFence = device->createFenceUnique(vk::FenceCreateInfo(vk::FenceCreateFlagBits::eSignaled));
    frameObjects[i].compressionFinishedEvent = device->createEventUnique(vk::EventCreateInfo());
    frameObjects[i].sumFinishedEvent = device->createEventUnique(vk::EventCreateInfo());
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

void AsyVkRender::createBufferUnique(vk::UniqueBuffer& buffer, vk::UniqueDeviceMemory& bufferMemory,
                                     vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties,
                                     vk::DeviceSize size)
{
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
  std::array<vk::DescriptorPoolSize, 11> poolSizes;

  poolSizes[0].type = vk::DescriptorType::eUniformBuffer;
  poolSizes[0].descriptorCount = options.maxFramesInFlight;

  poolSizes[1].type = vk::DescriptorType::eStorageBuffer;
  poolSizes[1].descriptorCount = options.maxFramesInFlight;

  poolSizes[2].type = vk::DescriptorType::eStorageBuffer;
  poolSizes[2].descriptorCount = options.maxFramesInFlight;

  poolSizes[3].type = vk::DescriptorType::eStorageBuffer;
  poolSizes[3].descriptorCount = options.maxFramesInFlight;

  poolSizes[4].type = vk::DescriptorType::eStorageBuffer;
  poolSizes[4].descriptorCount = options.maxFramesInFlight;

  poolSizes[5].type = vk::DescriptorType::eStorageBuffer;
  poolSizes[5].descriptorCount = options.maxFramesInFlight;

  poolSizes[6].type = vk::DescriptorType::eStorageBuffer;
  poolSizes[6].descriptorCount = options.maxFramesInFlight;

  poolSizes[7].type = vk::DescriptorType::eStorageBuffer;
  poolSizes[7].descriptorCount = options.maxFramesInFlight;

  poolSizes[8].type = vk::DescriptorType::eStorageBuffer;
  poolSizes[8].descriptorCount = options.maxFramesInFlight;

  poolSizes[9].type = vk::DescriptorType::eStorageBuffer;
  poolSizes[9].descriptorCount = options.maxFramesInFlight;

  poolSizes[10].type = vk::DescriptorType::eStorageBuffer;
  poolSizes[10].descriptorCount = options.maxFramesInFlight;

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
  auto allocInfo = vk::DescriptorSetAllocateInfo(
    *descriptorPool,
    VEC_VIEW(layouts)
  );
  auto descriptorSets = device->allocateDescriptorSetsUnique(allocInfo);

  for (size_t i = 0; i < options.maxFramesInFlight; i++)
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
  for (size_t i = 0; i < options.maxFramesInFlight; i++)
  {
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

    auto offsetBufferInfo = vk::DescriptorBufferInfo();

    offsetBufferInfo.buffer = *offsetBuffer;
    offsetBufferInfo.offset = 0;
    offsetBufferInfo.range = offsetBufferSize;

    auto opaqueBufferInfo = vk::DescriptorBufferInfo();

    opaqueBufferInfo.buffer = *opaqueBuffer;
    opaqueBufferInfo.offset = 0;
    opaqueBufferInfo.range = opaqueBufferSize;

    auto opaqueDepthBufferInfo = vk::DescriptorBufferInfo();

    opaqueDepthBufferInfo.buffer = *opaqueDepthBuffer;
    opaqueDepthBufferInfo.offset = 0;
    opaqueDepthBufferInfo.range = opaqueDepthBufferSize;

    auto indexBufferInfo = vk::DescriptorBufferInfo();

    indexBufferInfo.buffer = *indexBuffer;
    indexBufferInfo.offset = 0;
    indexBufferInfo.range = indexBufferSize;

    auto elementBufferInfo = vk::DescriptorBufferInfo();

    elementBufferInfo.buffer = *elementBuffer;
    elementBufferInfo.offset = 0;
    elementBufferInfo.range = elementBufferSize;

    std::array<vk::WriteDescriptorSet, 9> writes;

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

    writes[4].dstSet = *frameObjects[i].descriptorSet;
    writes[4].dstBinding = 4;
    writes[4].dstArrayElement = 0;
    writes[4].descriptorType = vk::DescriptorType::eStorageBuffer;
    writes[4].descriptorCount = 1;
    writes[4].pBufferInfo = &offsetBufferInfo;

    writes[5].dstSet = *frameObjects[i].descriptorSet;
    writes[5].dstBinding = 7;
    writes[5].dstArrayElement = 0;
    writes[5].descriptorType = vk::DescriptorType::eStorageBuffer;
    writes[5].descriptorCount = 1;
    writes[5].pBufferInfo = &opaqueBufferInfo;

    writes[6].dstSet = *frameObjects[i].descriptorSet;
    writes[6].dstBinding = 8;
    writes[6].dstArrayElement = 0;
    writes[6].descriptorType = vk::DescriptorType::eStorageBuffer;
    writes[6].descriptorCount = 1;
    writes[6].pBufferInfo = &opaqueDepthBufferInfo;

    writes[7].dstSet = *frameObjects[i].descriptorSet;
    writes[7].dstBinding = 9;
    writes[7].dstArrayElement = 0;
    writes[7].descriptorType = vk::DescriptorType::eStorageBuffer;
    writes[7].descriptorCount = 1;
    writes[7].pBufferInfo = &indexBufferInfo;

    writes[8].dstSet = *frameObjects[i].descriptorSet;
    writes[8].dstBinding = 10;
    writes[8].dstArrayElement = 0;
    writes[8].descriptorType = vk::DescriptorType::eStorageBuffer;
    writes[8].descriptorCount = 1;
    writes[8].pBufferInfo = &elementBufferInfo;

    device->updateDescriptorSets(writes.size(), writes.data(), 0, nullptr);
  }

  updateSceneDependentBuffers();

  // compute descriptors

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

void AsyVkRender::updateSceneDependentBuffers() {

  fragmentBufferSize = maxFragments*sizeof(glm::vec4);
  createBufferUnique(fragmentBuffer, fragmentBufferMemory,
                     vk::BufferUsageFlagBits::eStorageBuffer,
                     vk::MemoryPropertyFlagBits::eDeviceLocal,
                     fragmentBufferSize);

  depthBufferSize = maxFragments*sizeof(float);
  createBufferUnique(depthBuffer, depthBufferMemory,
                     vk::BufferUsageFlagBits::eStorageBuffer,
                     vk::MemoryPropertyFlagBits::eDeviceLocal,
                     depthBufferSize);

  for (size_t i = 0; i < options.maxFramesInFlight; i++) {

    auto fragmentBufferInfo = vk::DescriptorBufferInfo(
      *fragmentBuffer,
      0,
      fragmentBufferSize
    );
    auto depthBufferInfo = vk::DescriptorBufferInfo(
      *depthBuffer,
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

  createBufferUnique(feedbackBuffer,
                     feedbackBufferMemory,
                     vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
                     vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCached,
                     feedbackBufferSize);

  createBufferUnique(elementBuffer,
                     elementBufferMemory,
                     vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
                     vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCached,
                     elementBufferSize);

  for (size_t i = 0; i < options.maxFramesInFlight; i++) {

    createBufferUnique(frameObjects[i].uniformBuffer,
                       frameObjects[i].uniformBufferMemory,
                       vk::BufferUsageFlagBits::eUniformBuffer,
                       vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                       sizeof(UniformBufferObject));
    frameObjects[i].uboData = device->mapMemory(*frameObjects[i].uniformBufferMemory, 0, sizeof(UniformBufferObject), vk::MemoryMapFlags());
  }

  createDependentBuffers();
}

void AsyVkRender::createDependentBuffers()
{
  pixels=(swapChainExtent.width+1)*(swapChainExtent.height+1);
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

  vk::Flags<vk::MemoryPropertyFlagBits> countBufferFlags = vk::MemoryPropertyFlagBits::eDeviceLocal;

  if (!GPUindexing) {
    countBufferFlags = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCached;
  }

  if (countBufferMap || offsetStageBufferMap) {
    device->unmapMemory(*countBufferMemory);
    device->unmapMemory(*offsetStageBufferMemory);
  }

  createBufferUnique(countBuffer,
                     countBufferMemory,
                     vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc,
                     countBufferFlags,
                     countBufferSize);

  createBufferUnique(globalSumBuffer,
                     globalSumBufferMemory,
                     vk::BufferUsageFlagBits::eStorageBuffer,
                     vk::MemoryPropertyFlagBits::eDeviceLocal,
                     globalSize);

  createBufferUnique(offsetBuffer,
                     offsetBufferMemory,
                     vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
                     vk::MemoryPropertyFlagBits::eDeviceLocal,
                     offsetBufferSize);

  if (!GPUindexing) {
    createBufferUnique(offsetStageBuffer,
                       offsetStageBufferMemory,
                       vk::BufferUsageFlagBits::eTransferSrc,
                       vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCached,
                       offsetBufferSize);
  }

  createBufferUnique(opaqueBuffer,
                     opaqueBufferMemory,
                     vk::BufferUsageFlagBits::eStorageBuffer,
                     vk::MemoryPropertyFlagBits::eDeviceLocal,
                     opaqueBufferSize);

  createBufferUnique(opaqueDepthBuffer,
                     opaqueDepthBufferMemory,
                     vk::BufferUsageFlagBits::eStorageBuffer,
                     vk::MemoryPropertyFlagBits::eDeviceLocal,
                     opaqueBufferSize);

  createBufferUnique(indexBuffer,
                     indexBufferMemory,
                     vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
                     vk::MemoryPropertyFlagBits::eDeviceLocal,
                     indexBufferSize);

  if (!GPUindexing) {

    countBufferMap =  static_cast<std::uint32_t*>(device->mapMemory(*countBufferMemory, 0, countBufferSize));
    offsetStageBufferMap = static_cast<std::uint32_t*>(device->mapMemory(*offsetStageBufferMemory, 0, offsetBufferSize));
  }
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
  auto depthResolveAttachment = vk::AttachmentDescription2(
    vk::AttachmentDescriptionFlags(),
    vk::Format::eD32Sfloat,
    vk::SampleCountFlagBits::e1,
    vk::AttachmentLoadOp::eDontCare,
    vk::AttachmentStoreOp::eStore,
    vk::AttachmentLoadOp::eDontCare,
    vk::AttachmentStoreOp::eDontCare,
    vk::ImageLayout::eUndefined,
    vk::ImageLayout::eDepthStencilAttachmentOptimal
  );

  auto depthAttachmentRef = vk::AttachmentReference2(0, vk::ImageLayout::eDepthStencilAttachmentOptimal);
  auto depthResolveAttachmentRef = vk::AttachmentReference2(1, vk::ImageLayout::eDepthStencilAttachmentOptimal);

  auto depthResolveSubpass = vk::SubpassDescriptionDepthStencilResolve(
    vk::ResolveModeFlagBits::eMin,
    vk::ResolveModeFlagBits::eMax,
    &depthResolveAttachmentRef
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
    //&depthAttachmentRef,
    nullptr,
    0,
    nullptr,
    nullptr
    //&depthResolveSubpass
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
    //&depthResolveAttachmentRef,
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
    //&depthResolveAttachmentRef,
    nullptr,
    0,
    nullptr,
    nullptr
  );

  std::array<vk::AttachmentDescription2, 1> attachments
  {
    depthAttachment,
    //depthResolveAttachment
  };

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
    swapChainImageFormat,
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
    swapChainImageFormat,
    vk::SampleCountFlagBits::e1,
    vk::AttachmentLoadOp::eDontCare,
    vk::AttachmentStoreOp::eStore,
    vk::AttachmentLoadOp::eDontCare,
    vk::AttachmentStoreOp::eDontCare,
    vk::ImageLayout::eUndefined,
    vk::ImageLayout::ePresentSrcKHR
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
  auto depthResolveAttachment = vk::AttachmentDescription2(
    vk::AttachmentDescriptionFlags(),
    vk::Format::eD32Sfloat,
    vk::SampleCountFlagBits::e1,
    vk::AttachmentLoadOp::eDontCare,
    vk::AttachmentStoreOp::eStore,
    vk::AttachmentLoadOp::eDontCare,
    vk::AttachmentStoreOp::eDontCare,
    vk::ImageLayout::eUndefined,
    vk::ImageLayout::eDepthStencilAttachmentOptimal
  );

  auto colorAttachmentRef = vk::AttachmentReference2(0, vk::ImageLayout::eColorAttachmentOptimal);
  auto depthAttachmentRef = vk::AttachmentReference2(1, vk::ImageLayout::eDepthStencilAttachmentOptimal);
  auto depthResolveAttachmentRef = vk::AttachmentReference2(2, vk::ImageLayout::eDepthStencilAttachmentOptimal);
  auto colorResolveAttachmentRef = vk::AttachmentReference2(3, vk::ImageLayout::eColorAttachmentOptimal);

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
    //&depthResolveAttachmentRef
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
    depthResolveAttachment,
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

  auto dependency = vk::SubpassDependency2();

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

  options.emplace_back("HAVE_SSBO");

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
    static_cast<float>(swapChainExtent.width),
    static_cast<float>(swapChainExtent.height),
    0.0f,
    1.0f
  );

  auto scissor = vk::Rect2D(
    vk::Offset2D(0, 0),
    swapChainExtent
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

  auto const makePipeline = [=](vk::UniquePipeline & pipeline,
                                vk::PipelineShaderStageCreateInfo * stages,
                                vk::RenderPass renderPass,
                                std::uint32_t subpass)
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
      *graphicsPipelineLayout,
      renderPass,
      subpass,
      nullptr
    );

    if (auto result = device->createGraphicsPipelineUnique(nullptr, pipelineCI, nullptr);
        result.result != vk::Result::eSuccess)
      throw std::runtime_error("failed to create pipeline!");
    else
      pipeline = std::move(result.value);
  };

  makePipeline(
    graphicsPipeline,
    stages,
    type == PIPELINE_COUNT ? *countRenderPass : *graphicsRenderPass,
    graphicsSubpass
  );
}

void AsyVkRender::createGraphicsPipelines()
{
  auto const drawMode = options.mode == DRAWMODE_WIREFRAME ? vk::PolygonMode::eLine : vk::PolygonMode::eFill;

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
                        (PIPELINE_DONTCARE, compressPipeline, vk::PrimitiveTopology::eTriangleList,
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

  createImage(swapChainExtent.width, swapChainExtent.height, vk::SampleCountFlagBits::e1, vk::Format::eD32Sfloat,
              vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::MemoryPropertyFlagBits::eDeviceLocal, depthResolveImage,
              depthResolveImageMemory);
  createImageView(vk::Format::eD32Sfloat, vk::ImageAspectFlagBits::eDepth, depthResolveImage, depthResolveImageView);
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
  pushConstants.constants[2] = swapChainExtent.height;

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
    vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent),
    clearColors.size(),
    clearColors.data()
  );

  currentCommandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
}

void AsyVkRender::beginGraphicsFrameRender(int imageIndex)
{
  std::array<vk::ClearValue, 4> clearColors;

  clearColors[0] = vk::ClearValue(Background);
  clearColors[1].depthStencil.depth = 1.f;
  clearColors[1].depthStencil.stencil = 0;
  clearColors[2].depthStencil.depth = 1.f;
  clearColors[2].depthStencil.stencil = 0;
  clearColors[3] = vk::ClearValue(Background);

  auto renderPassInfo = vk::RenderPassBeginInfo(
    *graphicsRenderPass,
    *graphicsFramebuffers[imageIndex],
    vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent),
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

void AsyVkRender::recordCommandBuffer(DeviceBuffer & vertexBuffer,
                                      DeviceBuffer & indexBuffer,
                                      VertexBuffer * data,
                                      vk::UniquePipeline & pipeline,
                                      bool incrementRenderCount) {

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
  recordCommandBuffer(object.pointVertexBuffer,
                      object.pointIndexBuffer,
                      &pointData,
                      getPipelineType(pointPipelines));
  pointData.clear();
}

void AsyVkRender::drawLines(FrameObject & object)
{
  recordCommandBuffer(object.lineVertexBuffer,
                      object.lineIndexBuffer,
                      &lineData,
                      getPipelineType(linePipelines));
  lineData.clear();
}

void AsyVkRender::drawMaterials(FrameObject & object)
{
  recordCommandBuffer(object.materialVertexBuffer,
                      object.materialIndexBuffer,
                      &materialData,
                      getPipelineType(materialPipelines));
  materialData.clear();
}

void AsyVkRender::drawColors(FrameObject & object)
{
  recordCommandBuffer(object.colorVertexBuffer,
                      object.colorIndexBuffer,
                      &colorData,
                      getPipelineType(colorPipelines));
  colorData.clear();
}

void AsyVkRender::drawTriangles(FrameObject & object)
{
  recordCommandBuffer(object.triangleVertexBuffer,
                      object.triangleIndexBuffer,
                      &triangleData,
                      getPipelineType(trianglePipelines));
  triangleData.clear();
}

void AsyVkRender::drawTransparent(FrameObject & object)
{
  // todo has camp::ssbo
  recordCommandBuffer(object.transparentVertexBuffer,
                      object.transparentIndexBuffer,
                      &transparentData,
                      getPipelineType(transparentPipelines));

  transparentData.clear();
}

int ceilquotient(int x, int y)
{
  return (x+y-1)/y;
}

void AsyVkRender::partialSums(FrameObject & object, bool readSize)
{
  auto const writeBarrier = vk::MemoryBarrier( // todo sum2 fast
    vk::AccessFlagBits::eShaderWrite,
    vk::AccessFlagBits::eShaderRead
  );

  currentCommandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *sum1PipelineLayout, 0, 1, &*computeDescriptorSet, 0, nullptr);

  // run sum1
  currentCommandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eFragmentShader,
                                       vk::PipelineStageFlagBits::eComputeShader,
                                       { },
                                       1,
                                       &writeBarrier,
                                       0,
                                       nullptr,
                                       0,
                                       nullptr);
  currentCommandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *sum1Pipeline);
  currentCommandBuffer.dispatch(g, 1, 1);

  // run sum2
  auto const BlockSize=ceilquotient(g,localSize);
  currentCommandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                       vk::PipelineStageFlagBits::eComputeShader,
                                       { },
                                       1,
                                       &writeBarrier,
                                       0,
                                       nullptr,
                                       0,
                                       nullptr);
  currentCommandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *sum2Pipeline);
  currentCommandBuffer.pushConstants(*sum2PipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(std::uint32_t), &BlockSize);
  currentCommandBuffer.dispatch(1, 1, 1);

  // run sum3
  auto const Final=elements-1;
  currentCommandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                       vk::PipelineStageFlagBits::eComputeShader,
                                       { },
                                       1,
                                       &writeBarrier,
                                       0,
                                       nullptr,
                                       0,
                                       nullptr);
  currentCommandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *sum3Pipeline);
  currentCommandBuffer.pushConstants(*sum3PipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(std::uint32_t), &Final);
  currentCommandBuffer.dispatch(g, 1, 1);
  currentCommandBuffer.setEvent(*object.sumFinishedEvent, vk::PipelineStageFlagBits::eComputeShader);
}

std::uint32_t ceilpow2(std::uint32_t n)
{
  --n;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  return ++n;
}

void AsyVkRender::resizeBlendShader(std::uint32_t maxDepth) {

  maxSize=ceilpow2(maxDepth);
  recreateBlendPipeline=true;
}

void AsyVkRender::resizeFragmentBuffer(FrameObject & object) {

  if (GPUindexing) {

    vk::Result result;

    do
    {
      result = device->getEventStatus(*object.sumFinishedEvent);
    } while(result != vk::Result::eEventSet);

    static const auto feedbackMap=static_cast<std::uint32_t*>(device->mapMemory(*feedbackBufferMemory, 0, feedbackBufferSize, vk::MemoryMapFlags()));
    const auto maxDepth=feedbackMap[0];
    fragments=feedbackMap[1];

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

    currentCommandBuffer.fillBuffer(*countBuffer, 0, countBufferSize, 0);
  }

  beginCountFrameRender(imageIndex);
  currentCommandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *graphicsPipelineLayout, 0, 1, &*object.descriptorSet, 0, nullptr);

  if (!interlock) {

    recordCommandBuffer(object.pointVertexBuffer,
                        object.pointIndexBuffer,
                        &pointData,
                        getPipelineType(pointPipelines, true),
                        false);
    recordCommandBuffer(object.lineVertexBuffer,
                        object.lineIndexBuffer,
                        &lineData,
                        getPipelineType(linePipelines, true),
                        false);
    recordCommandBuffer(object.materialVertexBuffer,
                        object.materialIndexBuffer,
                        &materialData,
                        getPipelineType(materialPipelines, true),
                        false);
    recordCommandBuffer(object.colorVertexBuffer,
                        object.colorIndexBuffer,
                        &colorData,
                        getPipelineType(colorPipelines, true),
                        false);
    recordCommandBuffer(object.triangleVertexBuffer,
                        object.triangleIndexBuffer,
                        &triangleData,
                        getPipelineType(trianglePipelines, true),
                        false);
  }

  currentCommandBuffer.nextSubpass(vk::SubpassContents::eInline);

  // draw transparent
  recordCommandBuffer(object.transparentVertexBuffer,
                      object.transparentIndexBuffer,
                      &transparentData,
                      getPipelineType(transparentPipelines, true),
                      false);

  currentCommandBuffer.nextSubpass(vk::SubpassContents::eInline);

  if (GPUcompress) {

    static std::uint32_t* p = nullptr;

    if (p == nullptr) {
      p=static_cast<std::uint32_t*>(device->mapMemory(*elementBufferMemory, 0, elementBufferSize));
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

  if (GPUindexing) {

    beginFrameCommands(*object.computeCommandBuffer);
    g=ceilquotient(elements,groupSize);
    elements=groupSize*g;

    if(settings::verbose > 3) {
      static bool first=true;
      if(first) {
        partialSums(object);
        first=false;
      }
      unsigned int N=10000;
      utils::stopWatch Timer;
      for(unsigned int i=0; i < N; ++i)
        partialSums(object);

      // glFinish(); ??
      double T=Timer.seconds()/N;
      cout << "elements=" << elements << endl;
      cout << "Tmin (ms)=" << T*1e3 << endl;
      cout << "Megapixels/second=" << elements/T/1e6 << endl;
    }

    partialSums(object, true);
    endFrameCommands();
    commandsToSubmit.emplace_back(currentCommandBuffer);
  }

  auto info = vk::SubmitInfo();

  info.commandBufferCount = commandsToSubmit.size();
  info.pCommandBuffers = commandsToSubmit.data();

  renderQueue.submit(1, &info, *object.inComputeFence);

  if (!GPUindexing) { // todo transfer queue

    device->waitForFences(1, &*object.inComputeFence, true, std::numeric_limits<std::uint64_t>::max());

    elements=pixels;

    auto size=elements*sizeof(std::uint32_t);
    auto offset = offsetStageBufferMap+1;
    auto maxsize=countBufferMap[0];
    auto count=countBufferMap+1;
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
    object.copyCountCommandBuffer->copyBuffer(*offsetStageBuffer, *offsetBuffer, 1, &copy);
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

void AsyVkRender::drawBuffers(FrameObject & object, int imageIndex)
{
  copied=false;
  Opaque=transparentData.indices.empty();
  auto transparent=!Opaque;

  if (ssbo && transparent) {

    device->resetFences(1, &*object.inComputeFence);
    device->resetEvent(*object.sumFinishedEvent);
    device->resetEvent(*object.compressionFinishedEvent);

    object.countCommandBuffer->reset();
    object.computeCommandBuffer->reset();

    refreshBuffers(object, imageIndex);

    if (!interlock || true) {
      resizeFragmentBuffer(object);
    }
  }

  beginFrameCommands(getFrameCommandBuffer());

  if (!GPUindexing) {
    currentCommandBuffer.fillBuffer(*countBuffer, 0, countBufferSize, 0);
  }

  beginGraphicsFrameRender(imageIndex);
  currentCommandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *graphicsPipelineLayout, 0, 1, &*object.descriptorSet, 0, nullptr);
  drawPoints(object);
  drawLines(object);
  drawMaterials(object);
  drawColors(object);
  drawTriangles(object);

  currentCommandBuffer.nextSubpass(vk::SubpassContents::eInline);

  if (transparent) {

    drawTransparent(object);

    currentCommandBuffer.nextSubpass(vk::SubpassContents::eInline);

    blendFrame(imageIndex);

    // copied=true;

    // if (interlock) {
    //   resizeFragmentBuffer(object);
    // }
  }
  else {
    currentCommandBuffer.nextSubpass(vk::SubpassContents::eInline);
  }

  endFrameRender();
  endFrameCommands();

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

  std::array<vk::Fence, 2> fences {*frameObject.inFlightFence, *frameObject.inComputeFence};

  device->waitForFences(1, fences.data(), VK_TRUE, std::numeric_limits<uint64_t>::max());

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

  resetFrameCopyData();

  drawBuffers(frameObject, imageIndex);

  std::vector<vk::Semaphore> waitSemaphores {*frameObject.imageAvailableSemaphore};
  std::vector<vk::PipelineStageFlags> waitStages {vk::PipelineStageFlagBits::eColorAttachmentOutput};

  if (!GPUindexing) {

    waitSemaphores.emplace_back(*frameObject.inCountBufferCopy);
    waitStages.emplace_back(vk::PipelineStageFlagBits::eFragmentShader);
  }

  vk::Semaphore signalSemaphores[] = {*frameObject.renderFinishedSemaphore};
  auto submitInfo = vk::SubmitInfo(
    waitSemaphores.size(),
    waitSemaphores.data(),
    waitStages.data(),
    1,
    &*frameObject.commandBuffer,
    ARR_VIEW(signalSemaphores)
  );

  if (renderQueue.submit(1, &submitInfo, *frameObject.inFlightFence) != vk::Result::eSuccess)
    throw std::runtime_error("failed to submit draw command buffer!");

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
    {
      std::cout << "Other error: " << e.what() << std::endl;
      throw;
    }
  }

  if (recreateBlendPipeline) {

    device->waitForFences(1, &*frameObject.inFlightFence, VK_TRUE, std::numeric_limits<uint64_t>::max());
    createBlendPipeline();
    recreateBlendPipeline=false;
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
#ifdef HAVE_LIBGLFW
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
#ifdef HAVE_LIBGLFW
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

void AsyVkRender::windowposition(int& x, int& y, int width, int height)
{
  if (width==-1) {
    width=this->width;
  }
  if (height==-1) {
    height=this->height;
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

  if (reposition) {

    windowposition(x, y, w, h);
    glfwSetWindowPos(window, x, y);
  }
  else {

    glfwGetWindowPos(window, &x, &y);
    glfwSetWindowPos(window, max(x-2,0), max(y-2, 0));
  }

  glfwSetWindowSize(window, w, h);
  update();
}

void AsyVkRender::fullscreen(bool reposition) {

  width=screenWidth;
  height=screenHeight;

  if (firstFit) {

    if (width < height*aspect) {
      Zoom0 *= width/(height*aspect);
    }
    capzoom();
    setProjection();
    firstFit=false;
  }
  Xfactor=((double) screenHeight)/height;
  Yfactor=((double) screenWidth)/width;
  reshape0(width, height);
  if (reposition) {
    glfwSetWindowPos(window, 0, 0);
  }
  glfwSetWindowSize(window, width, height);
}

void AsyVkRender::reshape0(int width, int height) {

  X=(X/this->width)*width;
  Y=(Y/this->height)*height;

  this->width=width;
  this->height=height;

  static int lastWidth=1;
  static int lastHeight=1;
  if(View && this->width*this->height > 1 && (this->width != lastWidth || this->height != lastHeight)
     && settings::verbose > 1) {
    cout << "Rendering " << stripDir(Prefix) << " as "
         << this->width << "x" << this->height << " image" << endl;
    lastWidth=this->width;
    lastHeight=this->height;
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
      oldWidth=width;
      oldHeight=height;
      int w=screenWidth;
      int h=screenHeight;
      if(w > h*aspect)
        w=min((int) ceil(h*aspect),w);
      else
        h=min((int) ceil(w/aspect),h);

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
  x = y = cx = cy = 0;
  rotateMat = viewMat = glm::mat4(1.0);
  Zoom0 = 1.0;
  update();
}

void AsyVkRender::cycleMode() {
  options.mode = DrawMode((options.mode + 1) % DRAWMODE_MAX);
  recreatePipeline = true;
  remesh = true;
  redraw = true;
}

} // namespace camp
