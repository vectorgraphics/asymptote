#pragma once

#include <chrono>
#include <cmath>
#include <utility>
#include <memory>
#include <set>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <array>

#include "glmCommon.h"

#include "common.h"

#include "vk.h"
#ifdef HAVE_VULKAN
#include <vma_cxx.h>

#include <glslang/Public/ShaderLang.h>
#include <GLFW/glfw3.h>
#endif

#include "material.h"
#include "pen.h"
#include "triple.h"
#include "seconds.h"
#include "statistics.h"
#include "ThreadSafeQueue.h"
#include "vkRenderMessages.h"

#include "render.h"

namespace camp
{
class picture;

#define EMPTY_VIEW 0, nullptr
#define SINGLETON_VIEW(x) 1, &(x)
#define VEC_VIEW(x) static_cast<uint32_t>((x).size()), (x).data()
#define STD_ARR_VIEW(x) static_cast<uint32_t>((x).size()), (x).data()
#define ARR_VIEW(x) static_cast<uint32_t>(sizeof(x) / sizeof((x)[0])), x
#define RAW_VIEW(x) static_cast<uint32_t>(sizeof(x)), x
#define ST_VIEW(s) static_cast<uint32_t>(sizeof(s)), &s

template<class T>
inline T ceilquotient(T a, T b)
{
  return (a + b - 1) / b;
}

inline void store(float* f, double* C)
{
  f[0] = C[0];
  f[1] = C[1];
  f[2] = C[2];
}

inline void store(float* control, const triple& v)
{
  control[0] = v.getx();
  control[1] = v.gety();
  control[2] = v.getz();
}

inline void store(float* control, const triple& v, double weight)
{
  control[0] = v.getx() * weight;
  control[1] = v.gety() * weight;
  control[2] = v.getz() * weight;
  control[3] = weight;
}

std::vector<char> readFile(const std::string& filename);

enum DrawMode: int
{
   DRAWMODE_NORMAL,
   DRAWMODE_OUTLINE,
   DRAWMODE_WIREFRAME,
   DRAWMODE_MAX
};

struct Light
{
  glm::vec4 direction;
  glm::vec4 color;
};

#ifdef HAVE_VULKAN
struct SwapChainDetails {
  vk::SurfaceCapabilitiesKHR capabilities;
  std::vector<vk::SurfaceFormatKHR> formats;
  std::vector<vk::PresentModeKHR> presentModes;

  SwapChainDetails(vk::PhysicalDevice gpu, vk::SurfaceKHR surface);

  operator bool() const;

  vk::SurfaceFormatKHR chooseSurfaceFormat() const;
  vk::PresentModeKHR choosePresentMode() const;
  vk::Extent2D chooseExtent(size_t width, size_t height) const;
  std::uint32_t chooseImageCount() const;
};
#endif

struct QueueFamilyIndices {
  uint32_t transferQueueFamily;
  uint32_t renderQueueFamily;
  uint32_t presentQueueFamily;

  bool transferQueueFamilyFound = false;
  bool renderQueueFamilyFound = false;
  bool presentQueueFamilyFound = false;
};

struct UniformBufferObject {
  glm::mat4 projViewMat { };
  glm::mat4 viewMat { };
  glm::mat4 normMat { };
};

struct PushConstants
{
  glm::uvec4 constants;
  glm::vec4 background;
  // GRAPHICS:
    // constants[0] = flags
    // constants[1] = width
};

struct ComputePushConstants {
    uint32_t blockSize;
    uint32_t final;
};

struct Arcball {
  double angle;
  triple axis;

  Arcball(double x0, double y0, double x, double y)
  {
    triple v0 = norm(x0, y0);
    triple v1 = norm(x, y);
    double Dot = dot(v0, v1);
    angle = Dot > 1.0 ? 0.0 : Dot < -1.0 ? M_PI
                                         : acos(Dot);
    axis = unit(cross(v0, v1));
  }

  triple norm(double x, double y)
  {
    double norm = hypot(x, y);
    if (norm > 1.0) {
      double denom = 1.0 / norm;
      x *= denom;
      y *= denom;
    }
    return triple(x, y, sqrt(max(1.0 - x * x - y * y, 0.0)));
  }
};

struct projection
{
public:
  bool orthographic;
  camp::triple camera;
  camp::triple up;
  camp::triple target;
  double zoom;
  double angle;
  camp::pair viewportshift;

  projection(bool orthographic=false, camp::triple camera=0.0,
             camp::triple up=0.0, camp::triple target=0.0,
             double zoom=0.0, double angle=0.0,
             camp::pair viewportshift=0.0) :
    orthographic(orthographic), camera(camera), up(up), target(target),
    zoom(zoom), angle(angle), viewportshift(viewportshift) {}
};

#ifdef HAVE_VULKAN
constexpr
std::array<const char*, 4> deviceExtensions
{
  VK_KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME,
  VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME,
  VK_KHR_MULTIVIEW_EXTENSION_NAME,
  VK_KHR_MAINTENANCE2_EXTENSION_NAME
};

constexpr auto VB_USAGE_FLAGS = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer;
constexpr auto IB_USAGE_FLAGS = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer;
#endif

extern glm::dmat4 projViewMat;
extern glm::dmat4 normMat;

class AsyVkRender
{
public:

  AsyVkRender() = default;
  ~AsyVkRender();

  /** Argument for AsyVkRender::vkrender function */
  struct VkrenderFunctionArgs: public gc
  {
    string prefix;
    picture const* pic;
    string format;
    double width;
    double height;
    double angle;
    double zoom;
    triple m;
    triple M;
    pair shift;
    pair margin;

    double* t;
    double* tup;
    double* background;

    size_t nlightsin;

    triple* lights;
    double* diffuse;
    double* specular;

    bool view;
    int oldpid=0;
  };

  void vkrender(VkrenderFunctionArgs const& args);

  double getRenderResolution(triple Min) const;

  bool framebufferResized=false;
  bool recreatePipeline=false;
  bool recreateBlendPipeline=false;
  bool shouldUpdateBuffers=true;
  bool newUniformBuffer=true;
  bool queueExport=false;
  bool ibl=false;
  bool vkexit=false;
  bool hideWindow=false;

  bool vkthread=false;;
  bool initialize=true;
  bool copied=false;

  int maxFramesInFlight;
  size_t framecount;

  DrawMode mode = DRAWMODE_NORMAL;
  std::string title = "";

  /**
   * @remark Main thread is the consumer, other thread is the sender of messages;
   */
   ThreadSafeQueue<VulkanRendererMessage> messageQueue;

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

#ifdef HAVE_VULKAN
  vk::SampleCountFlagBits samples = vk::SampleCountFlagBits::e1;
#endif

  std::vector<Material> materials;
  MaterialMap materialMap;

  bool Opaque;
  std::uint32_t pixels;
  bool orthographic;

  glm::dmat4 rotateMat;
  glm::dmat4 projMat;
  glm::dmat4 viewMat;

  double xmin, xmax;
  double ymin, ymax;

  double Xmin, Xmax;
  double Ymin, Ymax;
  double Zmin, Zmax;

  int fullWidth, fullHeight;
  double X,Y;
  double Angle;
  double Zoom;
  double Zoom0;
  pair Shift;
  pair Margin;
  double ArcballFactor;

  camp::triple* Lights;
  double* LightsDiffuse;
  size_t nlights;
  std::array<float, 4> Background;

  const double* dprojView;
  const double* dView;

  double T[16];
  double Tup[16];

  void updateProjection();
  void frustum(double left, double right, double bottom,
               double top, double nearVal, double farVal);
  void ortho(double left, double right, double bottom,
             double top, double nearVal, double farVal);

  void clearCenters();
  void clearMaterials();

  bool redraw=false;
  bool redisplay=false;
  bool resize=false;
private:
#ifdef HAVE_VULKAN
  struct DeviceBuffer {
    vk::BufferUsageFlags usage;
    VkMemoryPropertyFlagBits properties;
    size_t nobjects;
    vk::DeviceSize stgBufferSize = 0;
    vma::cxx::UniqueBuffer _buffer;
    vma::cxx::UniqueBuffer _stgBuffer;

    DeviceBuffer(vk::BufferUsageFlags usage, VkMemoryPropertyFlagBits properties=VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        : usage(usage), properties(properties) {}

    void reset() {
      _buffer = vma::cxx::UniqueBuffer();
    }
  };
#endif

  const double pi=acos(-1.0);
  const double degrees=180.0/pi;
  const double radians=1.0/degrees;

  const picture* pic = nullptr;

  double H;
  double Xfactor, Yfactor;
  double cx, cy;

  int screenWidth, screenHeight;
  int width, height;
  int oldWidth,oldHeight;
  double Aspect;
  double oWidth, oHeight;
  double lastzoom;
  int Fitscreen=1;
  int Oldpid;

  utils::stopWatch spinTimer;
  utils::stopWatch fpsTimer;
  utils::stopWatch frameTimer;
  utils::statistics fpsStats;
  std::function<void()> currentIdleFunc = nullptr;
  bool Xspin = false;
  bool Yspin = false;
  bool Zspin = false;
  bool Animate = false;
  string Format;
  bool Step = false;
  bool View = false;
  string Prefix;
  bool ViewExport;
  bool antialias = false;
  bool readyAfterExport=false;

  bool remesh=true;
  bool interlock=false;
  bool GPUcompress=false;
  bool fxaa=false;
  bool srgb=false;

  vk::UniqueSemaphore renderTimelineSemaphore;
  uint64_t currentTimelineValue = 0;
  vk::UniqueSemaphore createTimelineSemaphore(uint64_t initialValue = 0);
  std::vector<vk::Semaphore> signalSemaphores;

#if defined(DEBUG)
  bool hasDebugMarker=false;
#endif

  std::int32_t gs2;
  std::int32_t gs;
  std::uint32_t g;
  std::uint32_t localSize;
  std::uint32_t blockSize;
  std::uint32_t groupSize;
  std::uint32_t elements;
  std::uint32_t fragments;
  std::uint32_t maxFragments;
  std::uint32_t maxSize=1;
  bool resetDepth=false;
  bool vkinitialize=true;

  size_t nmaterials=1; // Number of materials currently allocated in memory

#ifdef HAVE_VULKAN

  GLFWwindow* window=nullptr;
  vk::UniqueInstance instance;

  std::vector<const char*> validationLayers {};
  vk::UniqueSurfaceKHR surface;

  vk::PhysicalDevice physicalDevice = nullptr;
  vk::UniqueDevice device;

  vma::cxx::UniqueAllocator allocator;

  QueueFamilyIndices queueFamilyIndices;

  vk::Queue transferQueue;
  vk::Queue renderQueue;
  vk::Queue presentQueue;

  vk::UniqueSwapchainKHR swapChain;
  vk::UniqueCommandBuffer exportCommandBuffer;
  vk::UniqueFence exportFence;
  vk::Format backbufferImageFormat=vk::Format::eB8G8R8A8Unorm;
  vk::Format postProcFormat;
  vk::Extent2D backbufferExtent;
  vma::cxx::UniqueImage defaultBackbufferImg;
  std::vector<vk::Image> backbufferImages;
  std::vector<vk::UniqueImageView> backbufferImageViews;

#pragma region intermediate frame buffers
  std::vector<vma::cxx::UniqueImage> immediateRenderTargetImgs;
  std::vector<vk::UniqueImageView> immRenderTargetViews;
  std::vector<vk::UniqueSampler> immRenderTargetSampler;

  std::vector<vma::cxx::UniqueImage> prePresentationImages;
  std::vector<vk::UniqueImageView> prePresentationImgViews;
#pragma endregion
  std::vector<vk::UniqueFramebuffer> depthFramebuffers;
  std::vector<vk::UniqueFramebuffer> opaqueGraphicsFramebuffers;
  std::vector<vk::UniqueFramebuffer> graphicsFramebuffers;

  vma::cxx::UniqueImage depthImg;
  vk::UniqueImageView depthImageView;

  vma::cxx::UniqueImage depthResolveImg;
  vk::UniqueImageView depthResolveImageView;

  vk::SampleCountFlagBits msaaSamples;
  vma::cxx::UniqueImage colorImg;
  vk::UniqueImageView colorImageView;

  vk::UniqueCommandPool transferCommandPool;
  vk::UniqueCommandPool renderCommandPool;

  vk::UniqueDescriptorPool descriptorPool;

  vk::UniqueRenderPass countRenderPass;
  vk::UniqueRenderPass opaqueGraphicsRenderPass;
  vk::UniqueRenderPass graphicsRenderPass;
  vk::UniqueDescriptorSetLayout materialDescriptorSetLayout;

  vk::UniquePipelineLayout graphicsPipelineLayout;

  enum PipelineType
  {
    PIPELINE_OPAQUE,
    PIPELINE_TRANSPARENT,
    PIPELINE_COUNT,
    PIPELINE_MAX,
    PIPELINE_COMPRESS,
    PIPELINE_DONTCARE
  };
  std::vector<std::string> countShaderOptions {
  };
  std::vector<std::string> materialShaderOptions {
    "NORMAL"
  };
  std::vector<std::string> colorShaderOptions {
    "COLOR",
    "NORMAL"
  };
  std::vector<std::string> triangleShaderOptions {
    "COLOR",
    "NORMAL",
    "GENERAL"
  };
  std::vector<std::string> pointShaderOptions {
    "NOLIGHTS",
    "WIDTH"
  };
  std::vector<std::string> transparentShaderOptions {
    "COLOR",
    "NORMAL",
    "GENERAL",
    "TRANSPARENT"
  };

  vk::UniqueDebugUtilsMessengerEXT debugUtilsMsg;
  std::array<vk::UniquePipeline, PIPELINE_MAX> materialPipelines;
  std::array<vk::UniquePipeline, PIPELINE_MAX> transparentPipelines;
  std::array<vk::UniquePipeline, PIPELINE_MAX> trianglePipelines;
  std::array<vk::UniquePipeline, PIPELINE_MAX> linePipelines;
  std::array<vk::UniquePipeline, PIPELINE_MAX> pointPipelines;
  vk::UniquePipeline blendPipeline;
  vk::UniquePipeline compressPipeline;

  vk::UniqueDescriptorPool computeDescriptorPool;
  vk::UniqueDescriptorSetLayout computeDescriptorSetLayout;
  vk::UniqueDescriptorSet computeDescriptorSet;
  vk::UniquePipelineLayout sumPipelineLayout;
  vk::UniquePipeline sum1Pipeline;
  vk::UniquePipeline sum2Pipeline;
  vk::UniquePipeline sum3Pipeline;

  vma::cxx::UniqueBuffer materialBf;
  vma::cxx::UniqueBuffer lightBf;

  size_t countBufferSize;
  vma::cxx::UniqueBuffer countBf;

  size_t globalSize;
  vma::cxx::UniqueBuffer globalSumBf;

  size_t offsetBufferSize;
  vma::cxx::UniqueBuffer offsetBf;

  size_t feedbackBufferSize;
  vma::cxx::UniqueBuffer feedbackBf;

  size_t fragmentBufferSize;
  vma::cxx::UniqueBuffer fragmentBf;

  size_t depthBufferSize;
  vma::cxx::UniqueBuffer depthBf;

  size_t opaqueBufferSize;
  vma::cxx::UniqueBuffer opaqueBf;

  size_t opaqueDepthBufferSize;
  vma::cxx::UniqueBuffer opaqueDepthBf;

  size_t indexBufferSize;
  vma::cxx::UniqueBuffer indexBf;

  size_t elementBufferSize;
  vma::cxx::UniqueBuffer elementBf;
  std::unique_ptr<vma::cxx::MemoryMapperLock> elemBfMappedMem = nullptr;
  std::unique_ptr<vma::cxx::MemoryMapperLock> feedbackMappedPtr = nullptr;

  size_t transparencyCapacityPixels=0;

  vma::cxx::UniqueImage irradianceImg;
  vk::UniqueImageView irradianceView;
  vk::UniqueSampler irradianceSampler;

  vma::cxx::UniqueImage brdfImg;
  vk::UniqueImageView brdfView;
  vk::UniqueSampler brdfSampler;

  vma::cxx::UniqueImage reflectionImg;
  vk::UniqueImageView reflectionView;
  vk::UniqueSampler reflectionSampler;

#pragma region post-process compute stuff
  vk::Extent2D postProcessThreadGroupCount;

  vk::UniquePipeline postProcessPipeline;
  vk::UniquePipelineLayout postProcessPipelineLayout;
  vk::UniqueDescriptorSetLayout postProcessDescSetLayout;

  vk::UniqueDescriptorPool postProcessDescPool;

  std::vector<vk::UniqueDescriptorSet> postProcessDescSet;

  std::vector<vk::UniqueSemaphore> renderFinishedSemaphore;

#pragma endregion
  struct FrameObject {
    enum CommandBuffers {
      CMD_DEFAULT,
      CMD_COUNT,
      CMD_COMPUTE,
      CMD_COPY,
      CMD_PARTIAL,
      CMD_MAX
    };

    uint64_t timelineValue = 0;
    uint64_t computeTimelineValue = 0;
    vk::UniqueSemaphore imageAvailableSemaphore;
    vk::UniqueSemaphore inCountBufferCopy;
    vk::UniqueFence inFlightFence;
    vk::UniqueFence inComputeFence;
    vk::UniqueEvent compressionFinishedEvent;
    vk::UniqueEvent sumFinishedEvent;
    vk::UniqueEvent startTimedSumsEvent;
    vk::UniqueEvent timedSumsFinishedEvent;
    vk::UniqueSemaphore renderFinishedSemaphore;

    vk::UniqueCommandBuffer commandBuffer;
    vk::UniqueCommandBuffer countCommandBuffer;
    vk::UniqueCommandBuffer computeCommandBuffer;
    vk::UniqueCommandBuffer partialSumsCommandBuffer;
    vk::UniqueCommandBuffer copyCountCommandBuffer;

    vk::UniqueDescriptorSet descriptorSet;

    vma::cxx::UniqueBuffer uboBf;
    std::unique_ptr<vma::cxx::MemoryMapperLock> uboMappedMemory;

    vk::UniqueBuffer ssbo;
    vk::UniqueDeviceMemory ssboMemory;

    DeviceBuffer materialVertexBuffer = DeviceBuffer(VB_USAGE_FLAGS);
    DeviceBuffer materialIndexBuffer = DeviceBuffer(IB_USAGE_FLAGS);

    DeviceBuffer colorVertexBuffer = DeviceBuffer(VB_USAGE_FLAGS);
    DeviceBuffer colorIndexBuffer = DeviceBuffer(IB_USAGE_FLAGS);

    DeviceBuffer triangleVertexBuffer = DeviceBuffer(VB_USAGE_FLAGS);
    DeviceBuffer triangleIndexBuffer = DeviceBuffer(IB_USAGE_FLAGS);

    DeviceBuffer transparentVertexBuffer = DeviceBuffer(VB_USAGE_FLAGS);
    DeviceBuffer transparentIndexBuffer = DeviceBuffer(IB_USAGE_FLAGS);

    DeviceBuffer lineVertexBuffer = DeviceBuffer(VB_USAGE_FLAGS);
    DeviceBuffer lineIndexBuffer = DeviceBuffer(IB_USAGE_FLAGS);

    DeviceBuffer pointVertexBuffer = DeviceBuffer(VB_USAGE_FLAGS);
    DeviceBuffer pointIndexBuffer = DeviceBuffer(IB_USAGE_FLAGS);
#pragma region post-process compute stuff
    std::vector<vk::UniqueImage> resolvedColorImages;
    std::vector<vk::ImageView> resolveColorImgViews;
#pragma endregion

    void reset() {
        materialVertexBuffer.reset();
        colorVertexBuffer.reset();
        triangleVertexBuffer.reset();
        transparentVertexBuffer.reset();
        lineVertexBuffer.reset();
        pointVertexBuffer.reset();
    }
  };

  uint32_t currentFrame = 0;
  vk::CommandBuffer currentCommandBuffer;
  std::vector<FrameObject> frameObjects;
  std::string lastAction = "";

#endif

  void setDimensions(int Width, int Height, double X, double Y);
  void updateModelViewData();
  void setProjection();
  void update();

  static void updateHandler(int);

  static std::string getAction(int button, int mod);

#ifdef HAVE_VULKAN
  static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
  static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
  static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
  static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos);
  static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
  static void windowFocusCallback(GLFWwindow* window, int focused);

  void initWindow();
  void initVulkan();

  void createDebugMessenger();

  std::set<std::string> getInstanceExtensions();
  std::set<std::string> getDeviceExtensions(vk::PhysicalDevice& device);
  std::vector<const char*> getRequiredInstanceExtensions();
  void createInstance();
  void createSurface();
  void createAllocator();
  void pickPhysicalDevice();
  std::pair<std::uint32_t, vk::SampleCountFlagBits> getMaxMSAASamples( vk::PhysicalDevice& gpu );
  QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice& physicalDevice, vk::SurfaceKHR* surface);
  bool isDeviceSuitable(vk::PhysicalDevice& device);
  bool checkDeviceExtensionSupport(vk::PhysicalDevice& device);
  void createLogicalDevice();
  void transitionImageLayout(vk::CommandBuffer cmd,
                             vk::Image image,
			                       vk::AccessFlags srcAccessMask,
			                       vk::AccessFlags dstAccessMask,
			                       vk::ImageLayout oldImageLayout,
			                       vk::ImageLayout newImageLayout,
			                       vk::PipelineStageFlags srcStageMask,
			                       vk::PipelineStageFlags dstStageMask,
			                       vk::ImageSubresourceRange subresourceRange);
  void transitionFXAAImages();

  void createExportResources();
  void createSwapChain();
  void createOffscreenBuffers();
  void createImageViews();
  void createFramebuffers();
  void createCommandPools();
  void createCommandBuffers();
  void createImmediateRenderTargets();
  void setupPostProcessingComputeParameters();
  vk::CommandBuffer beginSingleCommands();
  void endSingleCommands(vk::CommandBuffer cmd);
  PushConstants buildPushConstants();
  vk::CommandBuffer & getFrameCommandBuffer();

  // Timeline semaphore helper functions
  void waitForTimelineSemaphore(vk::Semaphore semaphore, uint64_t value, uint64_t timeout = UINT64_MAX);
  vk::CommandBuffer & getFrameComputeCommandBuffer();
  vk::UniquePipeline & getPipelineType(std::array<vk::UniquePipeline, PIPELINE_MAX> & pipelines);
  void beginFrameCommands(vk::CommandBuffer cmd);
  void beginCountFrameRender(int imageIndex);
  void beginGraphicsFrameRender(int imageIndex);
  void resetFrameCopyData();
   void drawBuffer(DeviceBuffer & vertexBuffer,
                  DeviceBuffer & indexBuffer,
                  VertexBuffer * data,
                  vk::Pipeline pipeline,
                  bool incrementRenderCount=true);
  void postProcessImage(vk::CommandBuffer& cmdBuffer, uint32_t const& frameIndex);
  void copyToSwapchainImg(vk::CommandBuffer& cmdBuffer, uint32_t const& frameIndex);
  void endFrameRender();
  void endFrameCommands();
  void endFrame(int imageIndex);
  void createSyncObjects();
  void waitForEvent(vk::Event event);

  /**
   * Sets debug object name. This function is inert if compiling under release mode or if
   * hardware does not support Debug Markers
   * @param object Handle to a vulkan object, in uint64_t form
   * @param objType Debug object reporting type
   * @param name Name of the object to set
   */
  void setDebugObjectName(
          uint64_t const& object,
          vk::DebugReportObjectTypeEXT const& objType,
          std::string const& name
          );

  /**
   * Sets debug object name. This function is inert if compiling under release mode or if
   * hardware does not support Debug Markers
   * @tparam TVkObj Type of the Vulkan object. Requires debugReportType static constant.
   * @param object Handle to a vulkan object, under vk:: namespace
   * @param name Name of the object to set
   */
  template<typename TVkObj>
  void setDebugObjectName(
          TVkObj object,
          std::string const& name)
  {
    setDebugObjectName(
            reinterpret_cast<uint64_t>(static_cast<typename TVkObj::NativeType>(object)),
            TVkObj::debugReportObjectType,
            name);
  }

  uint32_t selectMemory(const vk::MemoryRequirements memRequirements, const vk::MemoryPropertyFlags properties);
  vma::cxx::UniqueBuffer createBufferUnique(
          vk::BufferUsageFlags const& usage,
          VkMemoryPropertyFlags const& properties,
          vk::DeviceSize const& size,
          VmaAllocationCreateFlags const& vmaFlags = 0,
          VmaMemoryUsage const& memoryUsage=VMA_MEMORY_USAGE_AUTO,
          const char* bufferName = nullptr);
  void copyBufferToBuffer(const vk::Buffer& srcBuffer, const vk::Buffer& dstBuffer, const vk::DeviceSize size);
  void copyToBuffer(
          const vk::Buffer& buffer,
          const void* data,
          vk::DeviceSize size,
          vma::cxx::UniqueBuffer const& stagingBuffer);
  void copyToBuffer(
          const vk::Buffer& buffer,
          const void* data,
          vk::DeviceSize size
  );

  static void zeroBuffer(vk::CommandBuffer const& cmdBuffer,
                         vk::Buffer const& buffer);

  void zeroTransparencyBuffers();
  vma::cxx::UniqueImage createImage(
    std::uint32_t w, std::uint32_t h, vk::SampleCountFlagBits samples, vk::Format fmt,
    vk::ImageUsageFlags usage, VkMemoryPropertyFlags props,
    vk::ImageType type=vk::ImageType::e2D, std::uint32_t depth=1
  );
  void createImageView(
          vk::Format fmt, vk::ImageAspectFlagBits flags, vk::Image const& img,
                       vk::UniqueImageView& imgView, vk::ImageViewType type=vk::ImageViewType::e2D);
  void createImageSampler(vk::UniqueSampler & sampler);
  void copyFromBuffer(const vk::Buffer& buffer, void* data, vk::DeviceSize size);
  void transitionImageLayout(vk::ImageLayout from, vk::ImageLayout to, vk::Image img);
  void copyDataToImage(const void *data, vk::DeviceSize size, vk::Image img,
                       std::uint32_t w, std::uint32_t h, vk::Offset3D const & offset={});
  void setDeviceBufferData(DeviceBuffer& buffer, const void* data, vk::DeviceSize size, size_t nobjects=0);

  void createDescriptorSetLayout();
  void createComputeDescriptorSetLayout();
  void createDescriptorPool();
  void createComputeDescriptorPool();
  void createDescriptorSets();
  void writeDescriptorSets();
  void writePostProcessDescSets();
  void writeMaterialAndLightDescriptors();
  void updateSceneDependentBuffers();

  void createMaterialVertexBuffer();
  void createMaterialIndexBuffer();

  void createBuffers();
  void createMaterialAndLightBuffers();
  void createDependentBuffers();

  void initIBL();

  void createCountRenderPass();
  void createGraphicsRenderPass();
  void createOpaqueRenderPass();
  void createTransparentRenderPass();
  void createBlendRenderPass();
  void createGraphicsPipelineLayout();
  void modifyShaderOptions(std::vector<std::string>& options, PipelineType type);
  template<typename V>
  void createGraphicsPipeline(PipelineType type, vk::UniquePipeline & graphicsPipeline, vk::PrimitiveTopology topology,
                              vk::PolygonMode fillMode, std::vector<std::string> options,
                              std::string const & name,
                              std::string const & vertexShader,
                              std::string const & fragmentShader,
                              int graphicsSubpass, bool enableDepthWrite=true,
                              bool transparent=false, bool disableMultisample=false);
  void createBlendPipeline();
  void createComputePipeline(
    vk::UniquePipelineLayout & layout,
    vk::UniquePipeline & pipeline,
    std::string const& shaderFile,
    std::vector<vk::DescriptorSetLayout> const& descSetLayout
  );
  void createComputePipelines();

  void createAttachments();

  void updateUniformBuffer(uint32_t currentImage);
  void updateBuffers();
  void drawPoints(FrameObject & object);
  void drawLines(FrameObject & object);
  void drawMaterials(FrameObject & object);
  void drawColors(FrameObject & object);
  void drawTriangles(FrameObject & object);
  void drawTransparent(FrameObject & object);
  void clearData();
  void partialSums(FrameObject & object, bool timing=false);
  void resizeBlendShader(std::uint32_t maxDepth);
  void resizeFragmentBuffer(FrameObject & object);
  void compressCount(FrameObject & object);
  void refreshBuffers(FrameObject & object, int imageIndex);
  void blendFrame(int imageIndex);
  void preDrawBuffers(FrameObject & object, int imageIndex);
  void drawBuffers(FrameObject & object, int imageIndex);
  void drawFrame();
  void recreateSwapChain();
  vk::UniqueShaderModule createShaderModule(EShLanguage lang, std::string const & filename, std::vector<std::string> const & options);
#endif

  void nextFrame();
  void clearBuffers();
  void render();
  void display();
  void mainLoop();
  void cleanup();
  void processMessages(VulkanRendererMessage const& msg);

  void idleFunc(std::function<void()> f);
  void idle();

  // user controls
  static void exportHandler(int=0);
  void Export(int imageIndex);
  bool readyForExport=false;
  bool readyForUpdate=false;
  bool waitEvent=true;
  bool initialized=false;
  bool havewindow=false;
  bool format3dWait=false;

  struct PipelineConfig {
    vk::PrimitiveTopology topology;
    vk::PolygonMode fillMode;
    std::vector<std::string>& shaderOptions;
    std::string namePrefix;
    std::string vertexShader;
    std::string fragmentShader;
    int graphicsSubpass;
    bool enableDepthWrite;
    bool transparent;
    bool disableMultisample;
  };

  template<typename V>
  void createGraphicsPipeline(PipelineType type, vk::UniquePipeline& graphicsPipeline, const PipelineConfig& config);
  void createGraphicsPipelines();

  template<typename V>
  void createPipelineSet(
    std::array<vk::UniquePipeline, PIPELINE_MAX>& pipelines,
    const PipelineConfig& config,
    PipelineType start = PIPELINE_OPAQUE,
    PipelineType end = PIPELINE_MAX
  );

  void quit();

  double spinStep();
  void rotateX(double step);
  void rotateY(double step);
  void rotateZ(double step);
  void xspin();
  void yspin();
  void zspin();
  void spinx();
  void spiny();
  void spinz();

  void animate();
  void expand();
  void shrink();
  projection camera(bool user=true);
  void showCamera();
  void shift(double dx, double dy);
  void pan(double dx, double dy);
  void capzoom();
  void zoom(double dx, double dy);
  void capsize(int& w, int& h);
  void windowposition(int& x, int& y, int width=-1, int height=-1);
  void setsize(int w, int h, bool reposition=true);
  void fullscreen(bool reposition=true);
  void reshape0(int width, int height);
  void setosize();
  void fitscreen(bool reposition=true);
  void toggleFitScreen();
  void home(bool webgl=false);
  void cycleMode();

  friend struct SwapChainDetails;
};

extern AsyVkRender* vk;

} // namespace camp
