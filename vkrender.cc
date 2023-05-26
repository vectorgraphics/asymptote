#include "vkrender.h"
#include "picture.h"

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

namespace gl {
bool glthread;
bool initialize;

#ifdef HAVE_PTHREAD
pthread_t mainthread;

pthread_cond_t initSignal = PTHREAD_COND_INITIALIZER;
pthread_mutex_t initLock = PTHREAD_MUTEX_INITIALIZER;

pthread_cond_t readySignal = PTHREAD_COND_INITIALIZER;
pthread_mutex_t readyLock = PTHREAD_MUTEX_INITIALIZER;

void endwait(pthread_cond_t& signal, pthread_mutex_t& lock)
{
  pthread_mutex_lock(&lock);
  pthread_cond_signal(&signal);
  pthread_mutex_unlock(&lock);
}
void wait(pthread_cond_t& signal, pthread_mutex_t& lock)
{
  pthread_mutex_lock(&lock);
  pthread_cond_signal(&signal);
  pthread_cond_wait(&signal,&lock);
  pthread_mutex_unlock(&lock);
}
#endif
}


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

// template<class T>
// void VertexQueue<T>::append(const VertexQueue<T>& other)
// {
//   appendOffset(indices, other.indices, vertices.size());
//   vertices.insert(vertices.end(), other.vertices.begin(), other.vertices.end());
//   materials.insert(materials.end(), other.materials.begin(), other.materials.end());
// }

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

  double cz = 0.5 * (Zmin + Zmax);
  viewMat = glm::translate(glm::translate(glm::dmat4(1.0), glm::dvec3(cx, cy, cz)) * rotateMat, glm::dvec3(0, 0, -cz));
  projViewMat = projMat * viewMat;
  // should this also be transposed? (would need to update billboardTransform)
  normMat = glm::inverse(viewMat);
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
  if (prerender <= 0.0) return 0.0;
  prerender = 1.0 / prerender;
  double perspective = orthographic ? 0.0 : 1.0 / Zmax;
  double s = perspective ? Min.getz() * perspective : 1.0;
  triple b(Xmin, Ymin, Zmin);
  triple B(Xmax, Ymin, Zmax);
  pair size3(s * (B.getx() - b.getx()), s * (B.gety() - b.gety()));
  // TODO: fullwidth, fullheight ?
  pair size2(width, height);
  return prerender * size3.length() / size2.length();
}

AsyVkRender::AsyVkRender(Options& options) : options(options)
{
  // VertexData2<MaterialVertex>::getBindingDescription();
  double pixelRatio = settings::getSetting<double>("devicepixelratio");

  if (this->options.display) {
    width = 800;
    height = 600;
    fullWidth = width;
    fullHeight = height;
    x = 0;
    y = 0;
    cx = 0;
    cy = 0;

    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    // remember last window position and size? (have as options?)
    window = glfwCreateWindow(width, height, options.title.data(), nullptr, nullptr);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);
  }
  initVulkan();
}

void AsyVkRender::framebufferResizeCallback(GLFWwindow* window, int width, int height)
{
  // can width or height be 0?
  auto app = reinterpret_cast<AsyVkRender*>(glfwGetWindowUserPointer(window));
  app->x = (app->x / app->width) * width;
  app->y = (app->y / app->height) * height;
  app->width = width;
  app->height = height;
  app->fullWidth = width;
  app->fullHeight = height;
  app->framebufferResized = true;
  app->redraw = true;
}

void AsyVkRender::scrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
  auto app = reinterpret_cast<AsyVkRender*>(glfwGetWindowUserPointer(window));
  app->Zoom0 *= yoffset > 0 ? (1. + 0.1 * yoffset) : 1 / (1 - 0.1 * yoffset);
  app->remesh = true;
  app->redraw = true;
}

void AsyVkRender::cursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{
  static double xprev = 0.0;
  static double yprev = 0.0;
  static bool first = true;

  if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS && !first) {
    auto app = reinterpret_cast<AsyVkRender*>(glfwGetWindowUserPointer(window));
    Arcball arcball(xprev * 2 / app->width - 1, 1 - yprev * 2 / app->height, xpos * 2 / app->width - 1, 1 - ypos * 2 / app->height);
    triple axis = arcball.axis;
    const double arcballFactor = 1.0;
    app->rotateMat = glm::rotate((2 * arcball.angle / app->Zoom0 * arcballFactor), glm::dvec3(axis.getx(), axis.gety(), axis.getz())) * app->rotateMat;
    app->redraw = true;
  }

  if (first) first = false;
  xprev = xpos;
  yprev = ypos;
}

AsyVkRender::~AsyVkRender()
{
  if (this->options.display) {
    glfwDestroyWindow(this->window);
    glfwTerminate();
  }
}

void AsyVkRender::vkrender(const picture* pic, const string& format,
                           double width, double height, double angle, double zoom,
                           const triple& mins, const triple& maxs, const pair& shift,
                           const pair& margin, double* t,
                           double* background, size_t nlightsin, triple* lights,
                           double* diffuse, double* specular, bool view)
{
  this->pic = pic;

  this->Angle = angle * M_PI / 180.0;
  this->Zoom0 = zoom;
  this->Shift = shift / zoom;
  this->Margin = margin;

  Xmin = mins.getx();
  Xmax = maxs.getx();
  Ymin = mins.gety();
  Ymax = maxs.gety();
  Zmin = mins.getz();
  Zmax = maxs.getz();

  orthographic = (this->Angle == 0.0);
  H = orthographic ? 0.0 : -tan(0.5 * this->Angle) * Zmax;
  xfactor = yfactor = 1.0;

  rotateMat = glm::mat4(1.0);
  viewMat = glm::mat4(1.0);

  // DeviceBuffer testBuffer(vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);
  // std::vector<uint8_t> data = {0, 1, 2, 3};
  // std::cout << "data size: " << data.size() << std::endl;
  // setDeviceBufferData(testBuffer, data.data(), data.size() * sizeof(data));
  // std::cout << "done setting device buffer" << std::endl;

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
  // createUniformBuffers();
  createBuffers();
  createDescriptorPool();
  createDescriptorSets();

  createMaterialRenderPass();
  createMaterialPipeline();
  createComputePipeline();

  createAttachments();

  createFramebuffers();
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
  createMaterialPipeline();
  createAttachments();
  createFramebuffers();
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

    auto const props = device.getProperties();

    std::string deviceType;

    if (vk::PhysicalDeviceType::eDiscreteGpu == props.deviceType)
      score += 10, deviceType = "discrete";
    else if (vk::PhysicalDeviceType::eIntegratedGpu == props.deviceType)
      score += 5, deviceType = "integrated";
    else if (vk::PhysicalDeviceType::eCpu == props.deviceType)
      deviceType = "cpu";
    else if (vk::PhysicalDeviceType::eVirtualGpu == props.deviceType)
      deviceType = "virtual";
    else if (vk::PhysicalDeviceType::eOther == props.deviceType)
      deviceType = "other";


    std::cout << "==================================================" << std::endl;
    std::cout << "Information for: " << props.deviceName << std::endl;
    std::cout << "\t type: " << deviceType << std::endl;
    std::cout << "\t max viewports: " << props.limits.maxViewports << std::endl;

    // todo uncomment
    if (!this->isDeviceSuitable(device))
      return std::cout << "NOT SUITABLE" << std::endl, score;

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

	if (count & vk::SampleCountFlagBits::e64)
		return vk::SampleCountFlagBits::e64;
	if (count & vk::SampleCountFlagBits::e32)
		return vk::SampleCountFlagBits::e32;
	if (count & vk::SampleCountFlagBits::e16)
		return vk::SampleCountFlagBits::e16;
	if (count & vk::SampleCountFlagBits::e8)
		return vk::SampleCountFlagBits::e8;
	if (count & vk::SampleCountFlagBits::e4)
		return vk::SampleCountFlagBits::e4;
	if (count & vk::SampleCountFlagBits::e2)
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

    if (family.queueFlags & vk::QueueFlagBits::eGraphics)
      indices.renderQueueFamily = u,
      indices.renderQueueFamilyFound = true;

    if (VK_FALSE != physicalDevice.getSurfaceSupportKHR(u, *surface))
      indices.presentQueueFamily = u,
      indices.presentQueueFamilyFound = true;

    if (family.queueFlags & vk::QueueFlagBits::eTransfer)
      indices.transferQueueFamily = u,
      indices.transferQueueFamilyFound = true;
  }

  return indices;
}

bool AsyVkRender::isDeviceSuitable(vk::PhysicalDevice& device)
{
  QueueFamilyIndices indices = findQueueFamilies(device, options.display ? &*surface : nullptr);

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
  std::cout << "\t extensions: " << std::endl;
  for (auto& extension : extensions) {
    std::cout << "\t\t" << extension.extensionName << std::endl;
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

  return details;
}

vk::SurfaceFormatKHR AsyVkRender::chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats)
{
  for (const auto& availableFormat : availableFormats) {
    if (availableFormat.format == vk::Format::eB8G8R8A8Srgb &&
        availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
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

  vk::SwapchainCreateInfoKHR swapchainCI = vk::SwapchainCreateInfoKHR(vk::SwapchainCreateFlagsKHR(), *surface, imageCount, surfaceFormat.format, surfaceFormat.colorSpace, extent, 1, vk::ImageUsageFlagBits::eColorAttachment, vk::SharingMode::eExclusive, 0, nullptr, swapChainSupport.capabilities.currentTransform, vk::CompositeAlphaFlagBitsKHR::eOpaque, presentMode, VK_TRUE, nullptr, nullptr);
  if (*swapChain)
    swapchainCI.oldSwapchain = *swapChain;

  if (queueFamilyIndices.renderQueueFamily != queueFamilyIndices.presentQueueFamily) {
    swapchainCI.imageSharingMode = vk::SharingMode::eConcurrent;
    swapchainCI.queueFamilyIndexCount = 2;
    swapchainCI.pQueueFamilyIndices=new uint32_t[2] {queueFamilyIndices.renderQueueFamily,queueFamilyIndices.presentQueueFamily};
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

// void AsyVkRender::copyFromBuffer(const vk::Buffer& buffer, void* data, vk::DeviceSize size,
//                                  bool wait = true, vk::Fence fence = {}, const vk::Semaphore semaphore = {},
//                                  vk::Buffer stagingBuffer = {}, vk::DeviceMemory stagingBufferMemory = {})
// {
//   vk::UniqueBuffer stagingBuffer;
//   vk::UniqueDeviceMemory stagingBufferMemory;
//   createBufferUnique(stagingBuffer, stagingBufferMemory, vk::BufferUsageFlagBits::eTransferDst,
//                      vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, size);

//   copyBufferToBuffer(buffer, *stagingBuffer, size);

//   void* memoryPtr = device->mapMemory(*stagingBufferMemory, 0, size, vk::MemoryMapFlags());
//   memcpy(data, memoryPtr, size);
//   device->unmapMemory(*stagingBufferMemory);
// }

void AsyVkRender::setDeviceBufferData(DeviceBuffer& buffer, const void* data, vk::DeviceSize size)
{
  // Vulkan doesn't allow a buffer to have a size of 0
  auto bufferCI = vk::BufferCreateInfo(vk::BufferCreateFlags(), std::max(vk::DeviceSize(1), size), buffer.usage);
  buffer.buffer = device->createBufferUnique(bufferCI);

  auto memRequirements = device->getBufferMemoryRequirements(*buffer.buffer);
  uint32_t memoryTypeIndex = selectMemory(memRequirements, buffer.properties);
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
  auto layoutCI = vk::DescriptorSetLayoutCreateInfo(
    vk::DescriptorSetLayoutCreateFlags(),
    1,
    &uboLayoutBinding
  );
  materialDescriptorSetLayout = device->createDescriptorSetLayoutUnique(layoutCI);
}

void AsyVkRender::createComputeDescriptorSetLayout()
{
  std::vector< vk::DescriptorSetLayoutBinding > layoutBindings
  {
    vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute),
    vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute),
    vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute)
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
  auto poolSize = vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, options.maxFramesInFlight);
  auto poolCI = vk::DescriptorPoolCreateInfo(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, options.maxFramesInFlight, 1, &poolSize);
  descriptorPool = device->createDescriptorPoolUnique(poolCI);
}

void AsyVkRender::createDescriptorSets()
{
  std::vector<vk::DescriptorSetLayout> layouts(options.maxFramesInFlight, *materialDescriptorSetLayout);
  auto allocInfo = vk::DescriptorSetAllocateInfo(*descriptorPool, VEC_VIEW(layouts));
  auto descriptorSets = device->allocateDescriptorSetsUnique(allocInfo);

  for (size_t i = 0; i < options.maxFramesInFlight; i++) {
    frameObjects[i].descriptorSet = std::move(descriptorSets[i]);
    auto bufferInfo = vk::DescriptorBufferInfo(*frameObjects[i].uniformBuffer, 0, sizeof(UniformBufferObject));
    auto descriptorWrite = vk::WriteDescriptorSet(*frameObjects[i].descriptorSet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &bufferInfo);
    device->updateDescriptorSets(1, &descriptorWrite, 0, nullptr);
  }
}

void AsyVkRender::createBuffers()
{
  for (size_t i = 0; i < options.maxFramesInFlight; i++) {
    // uniform buffer
    createBufferUnique(frameObjects[i].uniformBuffer,
                       frameObjects[i].uniformBufferMemory,
                       vk::BufferUsageFlagBits::eUniformBuffer,
                       vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                       sizeof(UniformBufferObject));
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

void AsyVkRender::createMaterialPipeline()
{
  auto vertShaderCode = readFile("shaders/material.vert.spv");
  auto fragShaderCode = readFile("shaders/material.frag.spv");

  vk::UniqueShaderModule vertShaderModule = createShaderModule(vertShaderCode);
  vk::UniqueShaderModule fragShaderModule = createShaderModule(fragShaderCode);

  vk::SpecializationMapEntry specializationMapEntries[] = {};
  uint32_t specializationData[] = {};
  auto specializationInfo = vk::SpecializationInfo(ARR_VIEW(specializationMapEntries), RAW_VIEW(specializationData));

  auto vertShaderStageCI = vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eVertex, *vertShaderModule, "main", &specializationInfo);
  auto fragShaderStageCI = vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eFragment, *fragShaderModule, "main", &specializationInfo);
  vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageCI, fragShaderStageCI};

  auto bindingDescription = MaterialVertex::getBindingDescription();
  auto attributeDescriptions = MaterialVertex::getAttributeDescriptions();
  auto vertexInputCI = vk::PipelineVertexInputStateCreateInfo(vk::PipelineVertexInputStateCreateFlags(), 1, &bindingDescription, VEC_VIEW(attributeDescriptions));

  auto inputAssemblyCI = vk::PipelineInputAssemblyStateCreateInfo(vk::PipelineInputAssemblyStateCreateFlags(), vk::PrimitiveTopology::eTriangleList, VK_FALSE);

  auto viewport = vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f);
  auto scissor = vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent);
  auto viewportStateCI = vk::PipelineViewportStateCreateInfo(vk::PipelineViewportStateCreateFlags(), 1, &viewport, 1, &scissor);

  // TODO: ask about frontface and cullmode
  auto rasterizerCI = vk::PipelineRasterizationStateCreateInfo(vk::PipelineRasterizationStateCreateFlags(), VK_FALSE, VK_FALSE, vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone, vk::FrontFace::eCounterClockwise, VK_FALSE, 0.0f, 0.0f, 0.0f, 1.0f);

  auto multisamplingCI = vk::PipelineMultisampleStateCreateInfo(vk::PipelineMultisampleStateCreateFlags(), msaaSamples, VK_FALSE, 0.0f, nullptr, VK_FALSE, VK_FALSE);

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

  auto pipelineLayoutCI = vk::PipelineLayoutCreateInfo(vk::PipelineLayoutCreateFlags(), 1, &*materialDescriptorSetLayout, 0, nullptr);

  materialPipelineLayout = device->createPipelineLayoutUnique(pipelineLayoutCI, nullptr);

  auto pipelineCI = vk::GraphicsPipelineCreateInfo(vk::PipelineCreateFlags(), ARR_VIEW(shaderStages), &vertexInputCI, &inputAssemblyCI, nullptr, &viewportStateCI, &rasterizerCI, &multisamplingCI, &depthStencilCI, &colorBlendCI, nullptr, *materialPipelineLayout, *materialRenderPass, 0, nullptr);

  if (auto result = device->createGraphicsPipelineUnique(nullptr, pipelineCI, nullptr);
      result.result != vk::Result::eSuccess)
    throw std::runtime_error("failed to create graphics pipeline!");
  else
    materialPipeline = std::move(result.value);
}

void AsyVkRender::createComputePipeline()
{
  auto computeShaderCode = readFile("shaders/compute.comp.spv");

  vk::UniqueShaderModule computeShaderModule = createShaderModule(computeShaderCode);

  auto computeShaderStageInfo = vk::PipelineShaderStageCreateInfo(
    vk::PipelineShaderStageCreateFlags(),
    vk::ShaderStageFlagBits::eCompute,
    *computeShaderModule,
    "main"
  );

  auto pipelineLayoutCI = vk::PipelineLayoutCreateInfo(
    vk::PipelineLayoutCreateFlags(),
    1,
    &*computeDescriptorSetLayout,
    0,
    nullptr
  );

  computePipelineLayout = device->createPipelineLayoutUnique(pipelineLayoutCI, nullptr);

  auto computePipelineCI = vk::ComputePipelineCreateInfo();

  computePipelineCI.layout = *computePipelineLayout;
  computePipelineCI.stage = computeShaderStageInfo;

  if (auto result = device->createComputePipelineUnique(VK_NULL_HANDLE, computePipelineCI);
      result.result != vk::Result::eSuccess)
    throw std::runtime_error("failed to create compute pipeline!");
  else
    computePipeline = std::move(result.value);
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
  UniformBufferObject ubo{ };
  // flip Y coordinate for Vulkan (Vulkan has different coordinate system than OpenGL)
  auto verticalFlipMat = glm::scale(glm::dmat4(1.0f), glm::dvec3(1.0f, -1.0f, 1.0f));
  ubo.projViewMat = verticalFlipMat * projViewMat;
  ubo.normMat = glm::mat3(glm::inverse(viewMat));

  auto data = device->mapMemory(*frameObjects[currentFrame].uniformBufferMemory, 0, sizeof(ubo), vk::MemoryMapFlags());
  memcpy(data, &ubo, sizeof(ubo));
  device->unmapMemory(*frameObjects[currentFrame].uniformBufferMemory);
}

void AsyVkRender::recordCommandBuffer(vk::CommandBuffer commandBuffer, uint32_t currentFrame, uint32_t imageIndex)
{
  auto beginInfo = vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eSimultaneousUse);
  commandBuffer.begin(beginInfo);
  std::array<vk::ClearValue, 2> clearColors;

  clearColors[0] = vk::ClearValue(std::array<float, 4>{0.0, 0.0, 0.0, 1.0});
  clearColors[1].depthStencil.depth = 1.f;
  clearColors[1].depthStencil.stencil = 0;

  auto renderPassInfo = vk::RenderPassBeginInfo(*materialRenderPass, *swapChainFramebuffers[imageIndex], vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent), clearColors.size(), &clearColors[0]);
  commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
  commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *materialPipeline);
  std::vector<vk::Buffer> vertexBuffers = {*frameObjects[currentFrame].materialVertexBuffer.buffer};
  std::vector<vk::DeviceSize> vertexOffsets = {0};
  commandBuffer.bindVertexBuffers(0, vertexBuffers, vertexOffsets);
  commandBuffer.bindIndexBuffer(*frameObjects[currentFrame].materialIndexBuffer.buffer, 0, vk::IndexType::eUint32);
  commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *materialPipelineLayout, 0, 1, &*frameObjects[currentFrame].descriptorSet, 0, nullptr);
  // TODO: we would need to guarantee that materialVertices and the buffers are synced or have another variable for this
  commandBuffer.drawIndexed(materialData.indices.size(), 1, 0, 0, 0);
  commandBuffer.endRenderPass();
  commandBuffer.end();
}

void AsyVkRender::drawFrame()
{
  auto& frameObject = frameObjects[currentFrame];

  // wait until this frame is finished before we start drawing the next one
  device->waitForFences(1, &*frameObject.inFlightFence, VK_TRUE, std::numeric_limits<uint64_t>::max());

  uint32_t imageIndex; // index of the current swap chain image to render to
  if (auto const result = device->acquireNextImageKHR(*swapChain, std::numeric_limits<uint64_t>::max(),
                                                      *frameObject.imageAvailableSemaphore, nullptr,
                                                      &imageIndex);
      result == vk::Result::eErrorOutOfDateKHR
      || result == vk::Result::eSuboptimalKHR)
    return recreateSwapChain();
  else if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR)
    throw std::runtime_error("Failed to acquire next swapchain image.");

  // signal this frame as in use
  device->resetFences(1, &*frameObject.inFlightFence);
  frameObject.commandBuffer->reset(vk::CommandBufferResetFlags());

  updateUniformBuffer(currentFrame);

  // optimize:
  //  - use semaphores instead of fences for vertex/index buffer upload
  //    - use locks (maybe a sperate task/thread) for waiting to release lock? (firgure out / do research)
  //  - smarter memroy management (make custom buffer class that acts like a vector?)

  // TODO: handle case with no vertices (Validation Error with allocationSize = 0)
  // TODO: check if the buffer actually needs to be updated

  // material vertex buffer
  // vk::DeviceSize materialVertexBufferSize = sizeof(materialVertices.vertices[0]) * materialVertices.vertices.size();
  // createBufferUnique(frameObject.materialVertexBuffer, frameObject.materialVertexBufferMemory, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
  //                    vk::MemoryPropertyFlagBits::eDeviceLocal, materialVertexBufferSize);
  // copyToBuffer(*frameObject.materialVertexBuffer, materialVertices.vertices.data(), materialVertexBufferSize);

  // // material index buffer
  // vk::DeviceSize materialIndexBufferSize = sizeof(materialVertices.indices[0]) * materialVertices.indices.size();
  // createBufferUnique(frameObject.materialIndexBuffer, frameObject.materialIndexBufferMemory, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
  //                    vk::MemoryPropertyFlagBits::eDeviceLocal, materialIndexBufferSize);
  // copyToBuffer(*frameObject.materialIndexBuffer, materialVertices.indices.data(), materialIndexBufferSize);

  // material vertex buffer
  setDeviceBufferData(frameObject.materialVertexBuffer, materialData.materialVertices.data(), materialData.materialVertices.size() * sizeof(materialData.materialVertices[0]));
  // material index buffer
  setDeviceBufferData(frameObject.materialIndexBuffer, materialData.indices.data(), materialData.indices.size() * sizeof(materialData.indices[0]));

  recordCommandBuffer(*frameObject.commandBuffer, currentFrame, imageIndex);

  vk::Semaphore waitSemaphores[] = {*frameObject.imageAvailableSemaphore};
  vk::PipelineStageFlags waitStages = vk::PipelineStageFlagBits::eColorAttachmentOutput;
  vk::Semaphore signalSemaphores[] = {*frameObject.renderFinishedSemaphore};
  auto submitInfo = vk::SubmitInfo(ARR_VIEW(waitSemaphores), &waitStages, 1, &*frameObject.commandBuffer, ARR_VIEW(signalSemaphores));

  if (renderQueue.submit(1, &submitInfo, *frameObject.inFlightFence) != vk::Result::eSuccess)
    throw std::runtime_error("failed to submit draw command buffer!");

  auto presentInfo = vk::PresentInfoKHR(ARR_VIEW(signalSemaphores), 1, &*swapChain, &imageIndex);

  try
  {
    if (auto const result = renderQueue.presentKHR(presentInfo);
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

  currentFrame = (currentFrame + 1) % options.maxFramesInFlight;
}

void AsyVkRender::display()
{
  setProjection();

  // what is this for?
  // if(remesh)
  //   camp::clearCenters();

  double perspective = orthographic ? 0.0 : 1.0 / Zmax;
  double diagonalSize = hypot(width, height);
  pic->render(diagonalSize, (xmin, ymin, Zmin), (xmax, ymax, Zmax), perspective, remesh);

  // createMaterialVertexBuffer();
  // createMaterialIndexBuffer();

  drawFrame();

  // TODO: why?
  if (!outlinemode) remesh = false;
}

void AsyVkRender::mainLoop()
{
  int i = 0;
  while (!glfwWindowShouldClose(window)) {
    // TODO: would we only need to rerender on a new event?
    // poll blocks until resizing is finished
    //  - would need to render on a separate thread to have smooth resizing
    // glfwPollEvents();
    glfwWaitEvents();

    if (framebufferResized) {
      recreateSwapChain();
    }

    if (redraw) {
      redraw = false;
      display();
    } else {
      // may not be needed if we are waiting for events
      // usleep(5000);
    }
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
  throw std::runtime_error("not implemented");
}

void AsyVkRender::clearMaterials()
{
  throw std::runtime_error("not implemented");
}

} // namespace camp
