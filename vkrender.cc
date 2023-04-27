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

namespace camp
{

std::vector<const char*> instanceExtensions = {
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
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
    throw std::runtime_error("failed to open file!");
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
  double xshift = (x / (double) width + shift.getx() * xfactor) * zoom;
  double yshift = (y / (double) height + shift.gety() * yfactor) * zoom;
  double zoominv = 1.0 / zoom;
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
  app->framebufferResized = true;
  app->redraw = true;
}

void AsyVkRender::scrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
  auto app = reinterpret_cast<AsyVkRender*>(glfwGetWindowUserPointer(window));
  app->zoom *= (1.0 + yoffset);
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
    app->rotateMat = glm::rotate((2 * arcball.angle / app->zoom * arcballFactor), glm::dvec3(axis.getx(), axis.gety(), axis.getz())) * app->rotateMat;
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
  std::cout << "vkrender" << std::endl;
  this->pic = pic;

  this->angle = angle * M_PI / 180.0;
  this->zoom = zoom;
  this->shift = shift / zoom;
  this->margin = margin;

  Xmin= mins.getx();
  Xmax = maxs.getx();
  Ymin = mins.gety();
  Ymax = maxs.gety();
  Zmin = mins.getz();
  Zmax = maxs.getz();

  orthographic = (this->angle == 0.0);
  H = orthographic ? 0.0 : -tan(0.5 * this->angle) * Zmax;
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
  // createUniformBuffers();
  createBuffers();
  createDescriptorPool();
  createDescriptorSets();

  createMaterialRenderPass();
  createMaterialPipeline();

  createFramebuffers();
}

void AsyVkRender::recreateSwapChain()
{
  int width = 0, height = 0;
  glfwGetFramebufferSize(window, &width, &height);
  // wait if the window is minimized
  while (width == 0 || height == 0) {
    glfwGetFramebufferSize(window, &width, &height);
    glfwWaitEvents();
  }
  framebufferResized = false;

  std::cout << "local width: " << width << " height: " << height << std::endl;
  std::cout << "this width: " << this->width << " height: " << this->height << std::endl;

  vkDeviceWaitIdle(*device);

  createSwapChain();
  createImageViews();
  createMaterialRenderPass();
  createMaterialPipeline();
  createFramebuffers();

  std::cout << "swap chain width: " << swapChainExtent.width << " height: " << swapChainExtent.height << std::endl;
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
  auto instanceCI = vk::InstanceCreateInfo(vk::InstanceCreateFlags(), &appInfo, VEC_VIEW(validationLayers), VEC_VIEW(extensions));
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
  auto devices = instance->enumeratePhysicalDevices();

  if (devices.size() == 0) {
    throw std::runtime_error("Failed to find a device with Vulkan support!");
  } else if (devices.size() == 1) {
    cout << "Using Device: " << devices[0].getProperties().deviceName << endl;
    physicalDevice = devices[0];
  } else {
    // TODO: pick best device if not specified
    if (options.device_index == -1) {
      cout << "Device Options:" << endl;
      for (size_t i = 0; i < devices.size(); i++) {
        cout << i << ": " << devices[i].getProperties().deviceName << endl;
        // devices[i].getProperties().deviceType;
      }
      cout << "Select Device: ";
      throw std::runtime_error("Device not specified!");
      // cin >> options.device_index;
    }
    if (options.device_index >= devices.size()) {
      throw std::runtime_error("Invalid device index");
    }
    cout << "Using Device: " << devices.at(options.device_index).getProperties().deviceName << endl;
    physicalDevice = devices.at(options.device_index);
  }
}


// maybe we should prefer using the same queue family for both transfer and render?
// TODO: use if instead of goto and favor same queue family
QueueFamilyIndices AsyVkRender::findQueueFamilies(vk::PhysicalDevice& physicalDevice, vk::SurfaceKHR* surface)
{
  QueueFamilyIndices indices;

  auto queueFamilies = physicalDevice.getQueueFamilyProperties();

  // find a queue family that supports graphics and compute for rendering OIT (preferred without presentation)
  for (int i = 0; i < queueFamilies.size(); i++) {
    auto queueFamily = queueFamilies[i];
    vk::Bool32 presentSupport = false;
    if (surface) presentSupport = physicalDevice.getSurfaceSupportKHR(i, *surface);
    if ((queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) && (queueFamily.queueFlags & vk::QueueFlagBits::eCompute) && !presentSupport) {
      indices.renderQueueFamily = i;
      indices.renderQueueFamilyFound = true;
      goto foundRenderQueueFamily;
    }
  }
  for (int i = 0; i < queueFamilies.size(); i++) {
    auto queueFamily = queueFamilies[i];
    if ((queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) && (queueFamily.queueFlags & vk::QueueFlagBits::eCompute)) {
      indices.renderQueueFamily = i;
      indices.renderQueueFamilyFound = true;
      goto foundRenderQueueFamily;
    }
  }
foundRenderQueueFamily:
  // find a queue family that supports presentation (preferably different from render queue family)
  if (surface) {
    for (int i = 0; i < queueFamilies.size(); i++) {
      vk::Bool32 presentSupport = physicalDevice.getSurfaceSupportKHR(i, *surface);
      if (presentSupport && i != indices.renderQueueFamily) {
        indices.presentQueueFamily = i;
        indices.presentQueueFamilyFound = true;
        goto foundPresentQueueFamily;
      }
    }
    for (int i = 0; i < queueFamilies.size(); i++) {
      vk::Bool32 presentSupport = physicalDevice.getSurfaceSupportKHR(i, *surface);
      if (presentSupport) {
        indices.presentQueueFamily = i;
        indices.presentQueueFamilyFound = true;
        goto foundPresentQueueFamily;
      }
    }
  }
foundPresentQueueFamily:
  // find a queue family that supports transfer (preferably different from other queue families)
  for (int i = 0; i < queueFamilies.size(); i++) {
    auto queueFamily = queueFamilies[i];
    if (queueFamily.queueFlags & vk::QueueFlagBits::eTransfer && i != indices.renderQueueFamily && (i != indices.presentQueueFamily || !indices.presentQueueFamilyFound)) {
      indices.transferQueueFamily = i;
      indices.transferQueueFamilyFound = true;
      goto foundTransferQueueFamily;
    }
  }
  for (int i = 0; i < queueFamilies.size(); i++) {
    auto queueFamily = queueFamilies[i];
    if (queueFamily.queueFlags & vk::QueueFlagBits::eTransfer) {
      indices.transferQueueFamily = i;
      indices.transferQueueFamilyFound = true;
      goto foundTransferQueueFamily;
    }
  }
foundTransferQueueFamily:

  return indices;
}


bool AsyVkRender::isDeviceSuitable(vk::PhysicalDevice& device)
{
  QueueFamilyIndices indices = findQueueFamilies(device, options.display ? &*surface : nullptr);

  bool extensionsSupported = checkDeviceExtensionSupport(device);

  bool swapChainAdequate = false;
  if (options.display && extensionsSupported) {
    SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device, *surface);
    swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
  }

  auto supportedFeatures = device.getFeatures();

  // TODO: check if we want anisotropy
  return indices.transferQueueFamilyFound && indices.renderQueueFamilyFound && (indices.presentQueueFamilyFound || !options.display) && extensionsSupported && (swapChainAdequate || !options.display) && supportedFeatures.samplerAnisotropy;
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
    hasExternalMemoryHostExtension = true;
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

SwapChainSupportDetails AsyVkRender::querySwapChainSupport(vk::PhysicalDevice& device, vk::SurfaceKHR& surface)
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

  std::cout << "swap chain minImageCount: " << swapChainSupport.capabilities.minImageCount << std::endl;
  uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
  if (swapChainSupport.capabilities.maxImageCount > 0 &&
      imageCount > swapChainSupport.capabilities.maxImageCount) {
    imageCount = swapChainSupport.capabilities.maxImageCount;
  }
  std::cout << "imageCount: " << imageCount << std::endl;

  vk::SwapchainCreateInfoKHR swapchainCI = vk::SwapchainCreateInfoKHR(vk::SwapchainCreateFlagsKHR(), *surface, imageCount, surfaceFormat.format, surfaceFormat.colorSpace, extent, 1, vk::ImageUsageFlagBits::eColorAttachment, vk::SharingMode::eExclusive, 0, nullptr, swapChainSupport.capabilities.currentTransform, vk::CompositeAlphaFlagBitsKHR::eOpaque, presentMode, VK_TRUE, nullptr, nullptr);

  if (queueFamilyIndices.renderQueueFamily != queueFamilyIndices.presentQueueFamily) {
    swapchainCI.imageSharingMode = vk::SharingMode::eConcurrent;
    swapchainCI.queueFamilyIndexCount = 2;
    swapchainCI.pQueueFamilyIndices = (uint32_t[]){queueFamilyIndices.renderQueueFamily, queueFamilyIndices.presentQueueFamily};
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
  for (size_t i = 0; i < swapChainImageViews.size(); i++) {
    vk::ImageView attachments[] = {*swapChainImageViews[i]};
    auto framebufferCI = vk::FramebufferCreateInfo(vk::FramebufferCreateFlags(), *materialRenderPass, ARR_VIEW(attachments), swapChainExtent.width, swapChainExtent.height, 1);
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
  auto uboLayoutBinding = vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment);
  auto layoutCI = vk::DescriptorSetLayoutCreateInfo(vk::DescriptorSetLayoutCreateFlags(), 1, &uboLayoutBinding);
  descriptorSetLayout = device->createDescriptorSetLayoutUnique(layoutCI);
}

void AsyVkRender::createDescriptorPool()
{
  auto poolSize = vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, options.maxFramesInFlight);
  auto poolCI = vk::DescriptorPoolCreateInfo(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, options.maxFramesInFlight, 1, &poolSize);
  descriptorPool = device->createDescriptorPoolUnique(poolCI);
}

void AsyVkRender::createDescriptorSets()
{
  std::vector<vk::DescriptorSetLayout> layouts(options.maxFramesInFlight, *descriptorSetLayout);
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
    createBufferUnique(frameObjects[i].uniformBuffer, frameObjects[i].uniformBufferMemory, vk::BufferUsageFlagBits::eUniformBuffer,
                       vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, sizeof(UniformBufferObject));
  }
}

void AsyVkRender::createMaterialRenderPass()
{
  auto colorAttachment = vk::AttachmentDescription(vk::AttachmentDescriptionFlags(), swapChainImageFormat, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::ePresentSrcKHR);

  auto colorAttachmentRef = vk::AttachmentReference(0, vk::ImageLayout::eColorAttachmentOptimal);

  auto subpass = vk::SubpassDescription(vk::SubpassDescriptionFlags(), vk::PipelineBindPoint::eGraphics, 0, nullptr, 1, &colorAttachmentRef);

  auto dependency = vk::SubpassDependency(VK_SUBPASS_EXTERNAL, 0, vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite, vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite);

  auto renderPassCI = vk::RenderPassCreateInfo(vk::RenderPassCreateFlags(), 1, &colorAttachment, 1, &subpass, 1, &dependency);

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

  auto multisamplingCI = vk::PipelineMultisampleStateCreateInfo(vk::PipelineMultisampleStateCreateFlags(), vk::SampleCountFlagBits::e1, VK_FALSE, 0.0f, nullptr, VK_FALSE, VK_FALSE);

  auto colorBlendAttachment = vk::PipelineColorBlendAttachmentState(VK_FALSE, vk::BlendFactor::eZero, vk::BlendFactor::eZero, vk::BlendOp::eAdd, vk::BlendFactor::eZero, vk::BlendFactor::eZero, vk::BlendOp::eAdd, vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);

  auto colorBlendCI = vk::PipelineColorBlendStateCreateInfo(vk::PipelineColorBlendStateCreateFlags(), VK_FALSE, vk::LogicOp::eCopy, 1, &colorBlendAttachment, {0.0f, 0.0f, 0.0f, 0.0f});

  auto pipelineLayoutCI = vk::PipelineLayoutCreateInfo(vk::PipelineLayoutCreateFlags(), 1, &*descriptorSetLayout, 0, nullptr);

  materialPipelineLayout = device->createPipelineLayoutUnique(pipelineLayoutCI, nullptr);

  auto pipelineCI = vk::GraphicsPipelineCreateInfo(vk::PipelineCreateFlags(), ARR_VIEW(shaderStages), &vertexInputCI, &inputAssemblyCI, nullptr, &viewportStateCI, &rasterizerCI, &multisamplingCI, nullptr, &colorBlendCI, nullptr, *materialPipelineLayout, *materialRenderPass, 0, nullptr);

  auto result = device->createGraphicsPipelineUnique(nullptr, pipelineCI, nullptr);
  if (result.result != vk::Result::eSuccess)
  {
    throw std::runtime_error("failed to create graphics pipeline!");
  }
  materialPipeline = std::move(result.value);
}

void AsyVkRender::updateUniformBuffer(uint32_t currentFrame)
{
  UniformBufferObject ubo{};
  // flip Y coordinate for Vulkan (Vulkan has different coordinate system than OpenGL)
  auto verticalFlipMat = glm::scale(glm::dmat4(1.0f), glm::dvec3(1.0f, -1.0f, 1.0f));
  ubo.projViewMat = verticalFlipMat * projViewMat;
  std::cout << glm::to_string(ubo.projViewMat) << std::endl;

  auto data = device->mapMemory(*frameObjects[currentFrame].uniformBufferMemory, 0, sizeof(ubo), vk::MemoryMapFlags());
  memcpy(data, &ubo, sizeof(ubo));
  device->unmapMemory(*frameObjects[currentFrame].uniformBufferMemory);
}

void AsyVkRender::recordCommandBuffer(vk::CommandBuffer commandBuffer, uint32_t currentFrame, uint32_t imageIndex)
{
  auto beginInfo = vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eSimultaneousUse);
  commandBuffer.begin(beginInfo);
  auto clearColor = vk::ClearValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f});
  auto renderPassInfo = vk::RenderPassBeginInfo(*materialRenderPass, *swapChainFramebuffers[imageIndex], vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent), 1, &clearColor);
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
  // signal this frame as in use
  device->resetFences(1, &*frameObject.inFlightFence);

  uint32_t imageIndex; // index of the current swap chain image to render to
  // wait until we have a good swapchain image
  while (true) {
    auto result = device->acquireNextImageKHR(*swapChain, std::numeric_limits<uint64_t>::max(), *frameObject.imageAvailableSemaphore, nullptr, &imageIndex);
    if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR) {
      recreateSwapChain();
      continue;
    } else if (result != vk::Result::eSuccess) {
      throw std::runtime_error("failed to present swap chain image!");
    } else {
      break;
    }
  }

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

  auto result = renderQueue.presentKHR(presentInfo);
  if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR || framebufferResized) {
    recreateSwapChain();
  } else if (result != vk::Result::eSuccess) {
    throw std::runtime_error("failed to present swap chain image!");
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

} // namespace camp
