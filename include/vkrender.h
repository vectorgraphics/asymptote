#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <utility>
#include <memory>
#include <set>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <array>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/transform.hpp>

#include "common.h"

#include "vk.h"
#ifdef HAVE_VULKAN
#include <vma_cxx.h>
#include <glslang/SPIRV/GlslangToSpv.h>
#include <GLFW/glfw3.h>
#endif

#include "material.h"
#include "pen.h"
#include "triple.h"
#include "seconds.h"
#include "statistics.h"
#include "ThreadSafeQueue.h"
#include "vkRenderMessages.h"

namespace camp
{

class picture;

// Comment out when not debugging:
#if defined(ENABLE_VK_VALIDATION)
#define VALIDATION
#endif

#define EMPTY_VIEW 0, nullptr
#define VEC_VIEW(x) static_cast<uint32_t>((x).size()), (x).data()
#define STD_ARR_VIEW(x) static_cast<uint32_t>((x).size()), (x).data()
#define ARR_VIEW(x) static_cast<uint32_t>(sizeof(x) / sizeof((x)[0])), x
#define RAW_VIEW(x) static_cast<uint32_t>(sizeof(x)), x
#define ST_VIEW(s) static_cast<uint32_t>(sizeof(s)), &s

typedef std::unordered_map<const Material, size_t> MaterialMap;

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

template<class T>
inline void extendOffset(std::vector<T>& a, const std::vector<T>& b, T offset)
{
  size_t n = a.size();
  size_t m = b.size();
  a.resize(n + m);
  for (size_t i = 0; i < m; ++i)
    a[n + i] = b[i] + offset;
}

/** Returns ceil(a/b) */
inline uint32_t integerDivRoundUp(uint32_t const& a, uint32_t const& b)
{
  // in float, this is a/b + (b-1)/b.
  // if b divides a, b-1/b is <0 and gets truncated
  // otherwise, let c = a mod b. Then a=kb + c where c < b
  // Then, a + b - 1 is kb + c + b - 1 = (k+1) + c - 1.
  // the (c-1)/b part gets truncated, leaving us with k + 1
  // which is what we want
  return (a + (b - 1)) / b;
}

std::vector<char> readFile(const std::string& filename);

#define POSITION_LOCATION 0
#define NORMAL_LOCATION   1
#define MATERIAL_LOCATION 2
#define COLOR_LOCATION    3
#define WIDTH_LOCATION    4

enum DrawMode: int
{
   DRAWMODE_NORMAL,
   DRAWMODE_OUTLINE,
   DRAWMODE_WIREFRAME,
   DRAWMODE_MAX
};

struct MaterialVertex
{
  glm::vec3 position;
  glm::vec3 normal;
  glm::i32 material;

#ifdef HAVE_VULKAN
  static vk::VertexInputBindingDescription getBindingDescription()
  {
    return vk::VertexInputBindingDescription(0, sizeof(MaterialVertex), vk::VertexInputRate::eVertex);
  }

  static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescriptions()
  {
    return std::array<vk::VertexInputAttributeDescription, 3>{
            vk::VertexInputAttributeDescription(POSITION_LOCATION, 0, vk::Format::eR32G32B32Sfloat, offsetof(MaterialVertex, position)),
            vk::VertexInputAttributeDescription(NORMAL_LOCATION, 0, vk::Format::eR32G32B32Sfloat, offsetof(MaterialVertex, normal)),
            vk::VertexInputAttributeDescription(MATERIAL_LOCATION, 0, vk::Format::eR32Sint, offsetof(MaterialVertex, material))};
  }
#endif
};

struct ColorVertex
{
  glm::vec3 position;
  glm::vec3 normal;
  glm::i32 material;
  glm::vec4 color;

#ifdef HAVE_VULKAN
  static vk::VertexInputBindingDescription getBindingDescription()
  {
    return vk::VertexInputBindingDescription(0, sizeof(ColorVertex), vk::VertexInputRate::eVertex);
  }

  static std::array<vk::VertexInputAttributeDescription, 4> getAttributeDescriptions()
  {
    return std::array<vk::VertexInputAttributeDescription, 4>{
            vk::VertexInputAttributeDescription(POSITION_LOCATION, 0, vk::Format::eR32G32B32Sfloat, offsetof(ColorVertex, position)),
            vk::VertexInputAttributeDescription(NORMAL_LOCATION, 0, vk::Format::eR32G32B32Sfloat, offsetof(ColorVertex, normal)),
            vk::VertexInputAttributeDescription(MATERIAL_LOCATION, 0, vk::Format::eR32Sint, offsetof(ColorVertex, material)),
            vk::VertexInputAttributeDescription(COLOR_LOCATION, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(ColorVertex, color))};
  }
#endif
};

struct PointVertex
{
  glm::vec3 position;
  glm::f32 width;
  glm::i32 material;

#ifdef HAVE_VULKAN
  static vk::VertexInputBindingDescription getBindingDescription()
  {
    return vk::VertexInputBindingDescription(0, sizeof(PointVertex), vk::VertexInputRate::eVertex);
  }

  static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescriptions()
  {
    return std::array<vk::VertexInputAttributeDescription, 3>{
            vk::VertexInputAttributeDescription(POSITION_LOCATION, 0, vk::Format::eR32G32B32Sfloat, offsetof(PointVertex, position)),
            vk::VertexInputAttributeDescription(WIDTH_LOCATION, 0, vk::Format::eR32Sfloat, offsetof(PointVertex, width)),
            vk::VertexInputAttributeDescription(MATERIAL_LOCATION, 0, vk::Format::eR32Sint, offsetof(PointVertex, material))};
  }
#endif
};

struct VertexBuffer {
  std::vector<MaterialVertex> materialVertices;
  std::vector<ColorVertex> colorVertices;
  std::vector<PointVertex> pointVertices;
  std::vector<std::uint32_t> indices;

  int renderCount=0;  // Are all patches in this buffer fully rendered?
  bool partial=false; // Does buffer contain incomplete data?
  bool copiedThisFrame=false;

  void clear()
  {
    materialVertices.clear();
    colorVertices.clear();
    pointVertices.clear();
    indices.clear();
  }

  std::uint32_t addVertex(const MaterialVertex& vertex)
  {
    std::uint32_t nvertices = materialVertices.size();
    materialVertices.push_back(vertex);
    return nvertices;
  }

  std::uint32_t addVertex(const ColorVertex& vertex)
  {
    std::uint32_t nvertices = colorVertices.size();
    colorVertices.push_back(vertex);
    return nvertices;
  }

  std::uint32_t addVertex(const PointVertex& vertex)
  {
    std::uint32_t nvertices = pointVertices.size();
    pointVertices.push_back(vertex);
    return nvertices;
  }

  void extendMaterial(const VertexBuffer& other)
  {
    extendOffset<std::uint32_t>(indices, other.indices, materialVertices.size());
    materialVertices.insert(materialVertices.end(), other.materialVertices.begin(), other.materialVertices.end());
  }

  void extendColor(const VertexBuffer& other)
  {
    extendOffset<std::uint32_t>(indices, other.indices, colorVertices.size());
    colorVertices.insert(colorVertices.end(), other.colorVertices.begin(), other.colorVertices.end());
  }

  void extendPoint(const VertexBuffer& other)
  {
    extendOffset<std::uint32_t>(indices, other.indices, pointVertices.size());
    pointVertices.insert(pointVertices.end(), other.pointVertices.begin(), other.pointVertices.end());
  }
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
  vk::Extent2D chooseExtent() const;
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

class AsyVkRender
{
public:

  AsyVkRender() = default;
  ~AsyVkRender();

  void vkrender(const string& prefix, const picture* pic, const string& format,
                double width, double height, double angle, double zoom,
                const triple& m, const triple& M, const pair& shift,
                const pair& margin, double* t,
                double* background, size_t nlightsin, triple* lights,
                double* diffuse, double* specular, bool view, int oldpid=0);

  triple billboardTransform(const triple& center, const triple& v) const;
  double getRenderResolution(triple Min) const;

  bool framebufferResized=false;
  bool recreatePipeline=false;
  bool recreateBlendPipeline=false;
  bool shouldUpdateBuffers=true;
  bool newUniformBuffer=true;
  bool queueExport=false;
  bool ibl=false;
  bool offscreen=false;
  bool vkexit=false;

  bool vkthread=false;;
  bool initialize=true;
  bool copied=false;

  int maxFramesInFlight;
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

  VertexBuffer materialData;
  VertexBuffer colorData;
  VertexBuffer triangleData;
  VertexBuffer transparentData;
  VertexBuffer lineData;
  VertexBuffer pointData;

  std::vector<Material> materials;
  MaterialMap materialMap;
  size_t materialIndex;

  unsigned int Opaque=0;
  std::uint32_t pixels;
  bool orthographic;

  glm::dmat4 rotateMat;
  glm::dmat4 projMat;
  glm::dmat4 viewMat;
  glm::dmat4 projViewMat;
  glm::dmat3 normMat;

  double xmin, xmax;
  double ymin, ymax;

  double Xmin, Xmax;
  double Ymin, Ymax;
  double Zmin, Zmax;

  int fullWidth, fullHeight;
  double X,Y;
  double Angle;
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

  double BBT[9];
  double T[16];

  void updateProjection();
  void frustum(double left, double right, double bottom,
               double top, double nearVal, double farVal);
  void ortho(double left, double right, double bottom,
             double top, double nearVal, double farVal);

  void clearCenters();
  void clearMaterials();

private:
#ifdef HAVE_VULKAN
  struct DeviceBuffer {
    vk::BufferUsageFlags usage;
    VkMemoryPropertyFlagBits properties;
    std::size_t nobjects;
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
  bool firstFit=true;
  double Aspect;
  double oWidth, oHeight;
  double lastZoom;
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
  bool exporting=false;

  bool remesh=true;
  bool redraw=true;
  bool interlock=false;
  bool GPUcompress=false;
  bool fxaa=false;
  bool srgb=false;

#if defined(DEBUG)
  bool hasDebugMarker=false;
#endif

  std::int32_t gs2;
  std::int32_t gs;
  std::uint32_t g;
  std::uint32_t processors;
  std::uint32_t localSize;
  std::uint32_t blockSize;
  std::uint32_t groupSize;
  std::uint32_t elements;
  std::uint32_t fragments;
  std::uint32_t maxFragments;
  std::uint32_t maxSize=1;

  size_t nmaterials=1; // Number of materials currently allocated in memory

#ifdef HAVE_VULKAN

  GLFWwindow* window=nullptr;
  vk::UniqueInstance instance;

#if defined(VALIDATION)
  vk::UniqueDebugUtilsMessengerEXT debugMessenger;
#endif
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
  std::vector<std::string> materialShaderOptions {
    "MATERIAL",
    "NORMAL"
  };
  std::vector<std::string> colorShaderOptions {
    "MATERIAL",
    "COLOR",
    "NORMAL"
  };
  std::vector<std::string> triangleShaderOptions {
    "MATERIAL",
    "COLOR",
    "NORMAL",
    "GENERAL"
  };
  std::vector<std::string> pointShaderOptions {
    "MATERIAL",
    "NOLIGHTS",
    "WIDTH"
  };
  std::vector<std::string> transparentShaderOptions {
    "MATERIAL",
    "COLOR",
    "NORMAL",
    "GENERAL",
    "TRANSPARENT"
  };

  std::array<vk::UniquePipeline, PIPELINE_MAX> materialPipelines;
  std::array<vk::UniquePipeline, PIPELINE_MAX> colorPipelines;
  std::array<vk::UniquePipeline, PIPELINE_MAX> transparentPipelines;
  std::array<vk::UniquePipeline, PIPELINE_MAX> trianglePipelines;
  std::array<vk::UniquePipeline, PIPELINE_MAX> linePipelines;
  std::array<vk::UniquePipeline, PIPELINE_MAX> pointPipelines;
  vk::UniquePipeline blendPipeline;
  vk::UniquePipeline compressPipeline;

  vk::UniqueDescriptorPool computeDescriptorPool;
  vk::UniqueDescriptorSetLayout computeDescriptorSetLayout;
  vk::UniqueDescriptorSet computeDescriptorSet;
  vk::UniquePipelineLayout sum1PipelineLayout;
  vk::UniquePipeline sum1Pipeline;
  vk::UniquePipelineLayout sum2PipelineLayout;
  vk::UniquePipeline sum2Pipeline;
  vk::UniquePipelineLayout sum3PipelineLayout;
  vk::UniquePipeline sum3Pipeline;

  vma::cxx::UniqueBuffer materialBf;
  vma::cxx::UniqueBuffer lightBf;

  std::size_t countBufferSize;
  vma::cxx::UniqueBuffer countBf;
  std::unique_ptr<vma::cxx::MemoryMapperLock> countBfMappedMem = nullptr;

  std::size_t globalSize;
  vma::cxx::UniqueBuffer globalSumBf;

  std::size_t offsetBufferSize;
  vma::cxx::UniqueBuffer offsetBf;
  vma::cxx::UniqueBuffer offsetStageBf;
  std::unique_ptr<vma::cxx::MemoryMapperLock> offsetStageBfMappedMem = nullptr;

  std::size_t feedbackBufferSize;
  vma::cxx::UniqueBuffer feedbackBf;

  std::size_t fragmentBufferSize;
  vma::cxx::UniqueBuffer fragmentBf;

  std::size_t depthBufferSize;
  vma::cxx::UniqueBuffer depthBf;

  std::size_t opaqueBufferSize;
  vma::cxx::UniqueBuffer opaqueBf;

  std::size_t opaqueDepthBufferSize;
  vma::cxx::UniqueBuffer opaqueDepthBf;

  std::size_t indexBufferSize;
  vma::cxx::UniqueBuffer indexBf;

  std::size_t elementBufferSize;
  vma::cxx::UniqueBuffer elementBf;

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

    vk::UniqueSemaphore imageAvailableSemaphore;
    vk::UniqueSemaphore renderFinishedSemaphore;
    vk::UniqueSemaphore inCountBufferCopy;
    vk::UniqueFence inFlightFence;
    vk::UniqueFence inComputeFence;
    vk::UniqueEvent compressionFinishedEvent;
    vk::UniqueEvent sumFinishedEvent;
    vk::UniqueEvent startTimedSumsEvent;
    vk::UniqueEvent timedSumsFinishedEvent;

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
  void updateViewmodelData();
  void setProjection();
  void update();

  static void updateHandler(int);

  static std::string getAction(int button, int mod);

#ifdef HAVE_VULKAN
  static void mouseButtonCallback(GLFWwindow * window, int button, int action, int mods);
  static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
  static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
  static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos);
  static void keyCallback(GLFWwindow * window, int key, int scancode, int action, int mods);

  void initWindow();
  void initVulkan();

#if defined(VALIDATION)
  void createDebugMessenger();
#endif

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
  vk::CommandBuffer & getFrameComputeCommandBuffer();
  vk::UniquePipeline & getPipelineType(std::array<vk::UniquePipeline, PIPELINE_MAX> & pipelines, bool count=false);
  void beginFrameCommands(vk::CommandBuffer cmd);
  void beginCountFrameRender(int imageIndex);
  void beginGraphicsFrameRender(int imageIndex);
  void resetFrameCopyData();
  void drawBuffer(DeviceBuffer & vertexBuffer,
                  DeviceBuffer & indexBuffer,
                  VertexBuffer * data,
                  vk::UniquePipeline & pipeline,
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
          VmaMemoryUsage const& memoryUsage=VMA_MEMORY_USAGE_AUTO);
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
  void setDeviceBufferData(DeviceBuffer& buffer, const void* data, vk::DeviceSize size, std::size_t nobjects=0);

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
                              std::string const & vertexShader,
                              std::string const & fragmentShader,
                              int graphicsSubpass, bool enableDepthWrite=true,
                              bool transparent=false, bool disableMultisample=false);
  void createGraphicsPipelines();
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
  void partialSums(FrameObject & object, bool timing=false, bool bindDescriptors=false);
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
  void display();
  optional<VulkanRendererMessage> poll();
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
  bool capsize(int& w, int& h);
  void windowposition(int& x, int& y, int width=-1, int height=-1);
  void setsize(int w, int h, bool reposition=true);
  void fullscreen(bool reposition=true);
  void reshape0(int width, int height);
  void setosize();
  void fitscreen(bool reposition=true);
  void toggleFitScreen();
  void travelHome(bool webgl=false);
  void cycleMode();

  friend class SwapChainDetails;
};

extern AsyVkRender* vk;

} // namespace camp
