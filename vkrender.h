#pragma once

// TODO: remove / move headers to .cc to avoid circular dependencies

// prevents errors, to remove later
#include <GL/glew.h>

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
#include <sys/time.h>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan.hpp>

#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/transform.hpp>

// #include "settings.h"

// #include "common.h"
// #include "locate.h"
// #include "seconds.h"
// #include "statistics.h"

// #include "pair.h"
// #include "picture.h"
// #include "triple.h"

#include "common.h"
#include "material.h"
#include "pen.h"
#include "triple.h"
#include "seconds.h"
#include "statistics.h"

/*
allow rendering output to file on systems without swapchain support

seperate function for present mode or flag?

seperate subclass for objects recreated every frame?

orthographic projection
*/

namespace camp
{

class picture;

// TODO: remove, add in makefile
#define DEBUG

#ifdef DEBUG
#define VALIDATION
#endif

#define VEC_VIEW(x) static_cast<uint32_t>(x.size()), x.data()
#define ARR_VIEW(x) static_cast<uint32_t>(sizeof(x) / sizeof(x[0])), x
#define RAW_VIEW(x) static_cast<uint32_t>(sizeof(x)), x
#define ST_VIEW(s) static_cast<uint32_t>(sizeof(s)), &s

typedef std::unordered_map<const Material, size_t> MaterialMap;

template<class T>
inline T ceilquotient(T a, T b)
{
  return (a + b - 1) / b;
}

inline void store(GLfloat* f, double* C)
{
  f[0] = C[0];
  f[1] = C[1];
  f[2] = C[2];
}

inline void store(GLfloat* control, const triple& v)
{
  control[0] = v.getx();
  control[1] = v.gety();
  control[2] = v.getz();
}

inline void store(GLfloat* control, const triple& v, double weight)
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

std::vector<char> readFile(const std::string& filename);

#define POSITION_LOCATION 0
#define NORMAL_LOCATION   1
#define MATERIAL_LOCATION 2
#define COLOR_LOCATION    3
#define WIDTH_LOCATION    4

struct MaterialVertex // vertexData
{
  glm::vec3 position;
  glm::vec3 normal;
  glm::i32 material;

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
};

struct ColorVertex // VertexData
{
  glm::vec3 position;
  glm::vec3 normal;
  glm::i32 material;
  glm::vec4 color;

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
};

struct PointVertex // vertexData0
{
  glm::vec3 position;
  glm::f32 width;
  glm::i32 material;

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
};

struct VertexBuffer {
  std::vector<MaterialVertex> materialVertices;
  std::vector<ColorVertex> colorVertices;
  std::vector<PointVertex> pointVertices;
  std::vector<GLuint> indices;

  int renderCount = 0;
  bool partial = false;
  bool copiedThisFrame=false;

  void clear()
  {
    materialVertices.clear();
    colorVertices.clear();
    pointVertices.clear();
    indices.clear();
  }

  GLuint addVertex(const MaterialVertex& vertex)
  {
    GLuint nvertices = materialVertices.size();
    materialVertices.push_back(vertex);
    return nvertices;
  }

  GLuint addVertex(const ColorVertex& vertex)
  {
    GLuint nvertices = colorVertices.size();
    colorVertices.push_back(vertex);
    return nvertices;
  }

  GLuint addVertex(const PointVertex& vertex)
  {
    GLuint nvertices = pointVertices.size();
    pointVertices.push_back(vertex);
    return nvertices;
  }

  void extendMaterial(const VertexBuffer& other)
  {
    extendOffset<GLuint>(indices, other.indices, materialVertices.size());
    materialVertices.insert(materialVertices.end(), other.materialVertices.begin(), other.materialVertices.end());
  }

  void extendColor(const VertexBuffer& other)
  {
    extendOffset<GLuint>(indices, other.indices, colorVertices.size());
    colorVertices.insert(colorVertices.end(), other.colorVertices.begin(), other.colorVertices.end());
  }

  void extendPoint(const VertexBuffer& other)
  {
    extendOffset<GLuint>(indices, other.indices, pointVertices.size());
    pointVertices.insert(pointVertices.end(), other.pointVertices.begin(), other.pointVertices.end());
  }
};

struct Light
{
  glm::vec4 direction;
  glm::vec4 color;
};

struct SwapChainSupportDetails {
  vk::SurfaceCapabilitiesKHR capabilities;
  std::vector<vk::SurfaceFormatKHR> formats;
  std::vector<vk::PresentModeKHR> presentModes;
};

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

enum DrawMode: int
{
   DRAWMODE_NORMAL,
   DRAWMODE_OUTLINE,
   DRAWMODE_WIREFRAME,
   DRAWMODE_MAX
};

class AsyVkRender
{
public:

  struct Options {
    DrawMode mode = DRAWMODE_NORMAL;
    bool display = false;
    std::string title = "";
    int maxFramesInFlight = 5;
    vk::PresentModeKHR presentMode = vk::PresentModeKHR::eImmediate; //vk::PresentModeKHR::eFifo;
    vk::SampleCountFlagBits samples = vk::SampleCountFlagBits::e1;

    Options() = default;
  };

  Options options;

  AsyVkRender() = default;
  ~AsyVkRender();

  void vkrender(const picture* pic, const string& format,
                double width, double height, double angle, double zoom,
                const triple& m, const triple& M, const pair& shift,
                const pair& margin, double* t,
                double* background, size_t nlightsin, triple* lights,
                double* diffuse, double* specular, bool view, int oldpid=0);

  triple billboardTransform(const triple& center, const triple& v) const;
  double getRenderResolution(triple Min) const;

  bool framebufferResized = false;
  bool recreatePipeline = false;
  bool updateLights = true;
  bool newUniformBuffer = true;
  bool queueExport = false;
  bool format3dWait = false;
  bool vkexit = false;

  bool vkthread=false;;
  bool initialize=true;
  bool vkinit=false;
  bool copied=false;

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

  VertexBuffer materialData;
  VertexBuffer colorData;
  VertexBuffer triangleData;
  VertexBuffer transparentData;
  VertexBuffer lineData;
  VertexBuffer pointData;

  std::vector<Material> materials;
  std::vector<Material> oldMaterials;
  MaterialMap materialMap;
  size_t materialIndex;

  unsigned int Opaque=0;
  bool outlinemode = false;
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
  double Angle;
  double Zoom0;
  pair Shift;
  pair Margin;
  double ArcballFactor;

  camp::triple* Lights;
  double* LightsDiffuse;
  size_t nlights;
  double* Diffuse;
  std::array<float, 4> Background;

  const double* dprojView;
  const double* dView;

  double BBT[9];
  double T[16];

  size_t Nmaterials;  
  size_t nmaterials;  
  size_t Maxmaterials;

  void updateProjection();
  void frustum(GLdouble left, GLdouble right, GLdouble bottom,
               GLdouble top, GLdouble nearVal, GLdouble farVal);
  void ortho(GLdouble left, GLdouble right, GLdouble bottom,
             GLdouble top, GLdouble nearVal, GLdouble farVal);

  void clearCenters();
  void clearMaterials();

private:
  struct DeviceBuffer {
    vk::BufferUsageFlags usage;
    vk::MemoryPropertyFlags properties;
    vk::DeviceSize memorySize = 0;
    std::size_t nobjects;
    vk::UniqueBuffer buffer;
    vk::UniqueDeviceMemory memory;
    // only used if hasExternalMemoryHostExtension == false
    vk::UniqueBuffer stagingBuffer;
    vk::UniqueDeviceMemory stagingBufferMemory;

    DeviceBuffer(vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties)
        : usage(usage), properties(properties) {}
  };

  const picture* pic = nullptr;

  double H;
  double xfactor, yfactor; // what is this for?

  double x, y; // make these more descriptive (something to do with shift?)

  double cx, cy; // center variables

  int screenWidth, screenHeight;
  int width, height; // width and height of the window
  double aspect;
  double oWidth, oHeight;
  double lastZoom;
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
  bool queueScreen = false;
  string Format;
  bool Step = false;
  bool View = false;
  string Prefix;
  bool ViewExport;
  bool antialias = false;
  bool readyAfterExport=false;
  bool exporting=false;

  bool remesh=true; // whether picture needs to be remeshed
  bool redraw=true; // whether a new frame needs to be rendered
  bool ssbo=true;
  bool interlock=false;
  bool GPUindexing=false;
  bool GPUcompress=false;
  bool initSSBO=true;

  GLint gs2;
  GLint gs;
  GLint g;
  GLuint processors;
  GLuint localSize;
  GLuint blockSize;
  GLuint groupSize;
  GLuint elements;
  GLuint fragments;
  GLuint maxFragments=1;
  //GLint maxgroups;
  GLuint maxSize=1;

  bool hasExternalMemoryCapabilitiesExtension = false;
  bool hasExternalMemoryExtension = false;
  bool hasExternalMemoryHostExtension = false;

  size_t NMaterials = 48;

  GLFWwindow* window;

  vk::UniqueInstance instance;
  vk::UniqueSurfaceKHR surface;

  vk::PhysicalDevice physicalDevice = nullptr;
  vk::UniqueDevice device;

  QueueFamilyIndices queueFamilyIndices;

  // TODO: test and think about case where all queues are same family
  vk::Queue transferQueue; // not needed?
  vk::Queue renderQueue;   // supports both graphics and compute for OIT rendering (guaranteed to be available by Vulkan spec)
  vk::Queue presentQueue;  // prefer separate for no good reason

  vk::UniqueSwapchainKHR swapChain;
  vk::UniqueCommandBuffer exportCommandBuffer;
  vk::UniqueFence exportFence;
  std::vector<vk::Image> swapChainImages;
  vk::Format swapChainImageFormat;
  vk::Extent2D swapChainExtent;
  std::vector<vk::UniqueImageView> swapChainImageViews;
  std::vector<vk::UniqueFramebuffer> depthFramebuffers;
  std::vector<vk::UniqueFramebuffer> graphicsFramebuffers;

  vk::UniqueImage depthImage;
  vk::UniqueImageView depthImageView;
  vk::UniqueDeviceMemory depthImageMemory;

  vk::UniqueImage depthResolveImage;
  vk::UniqueImageView depthResolveImageView;
  vk::UniqueDeviceMemory depthResolveImageMemory;

  vk::SampleCountFlagBits msaaSamples;
  vk::UniqueImage colorImage;
  vk::UniqueImageView colorImageView;
  vk::UniqueDeviceMemory colorImageMemory;

  vk::UniqueCommandPool transferCommandPool;
  vk::UniqueCommandPool renderCommandPool;

  vk::UniqueDescriptorPool descriptorPool;

  vk::UniqueRenderPass countRenderPass;
  vk::UniqueRenderPass graphicsRenderPass;
  vk::UniqueDescriptorSetLayout materialDescriptorSetLayout;

  vk::UniquePipelineLayout graphicsPipelineLayout;

  enum PipelineType
  {
    PIPELINE_OPAQUE,
    PIPELINE_COUNT,
    PIPELINE_INDEXING,
    PIPELINE_INDEXING_SSBO,
    PIPELINE_MAX
  };
  std::string const shaderExtensions[PIPELINE_MAX] =
  {
    "Opaque",
    "Count",
    "Indexing",
    "IndexingSSBO"
  };

  std::array<vk::UniquePipeline, PIPELINE_MAX> materialPipelines;
  std::array<vk::UniquePipeline, PIPELINE_MAX> colorPipelines;
  std::array<vk::UniquePipeline, PIPELINE_MAX> transparentPipelines;
  std::array<vk::UniquePipeline, PIPELINE_MAX> trianglePipelines;
  std::array<vk::UniquePipeline, PIPELINE_MAX> linePipelines;
  std::array<vk::UniquePipeline, PIPELINE_MAX> pointPipelines;
  std::array<vk::UniquePipeline, PIPELINE_MAX> blendPipelines;

  vk::UniqueDescriptorPool computeDescriptorPool;
  vk::UniqueDescriptorSetLayout computeDescriptorSetLayout;
  vk::UniqueDescriptorSet computeDescriptorSet;
  vk::UniquePipelineLayout sum1PipelineLayout;
  vk::UniquePipeline sum1Pipeline;
  vk::UniquePipelineLayout sum2PipelineLayout;
  vk::UniquePipeline sum2Pipeline;
  vk::UniquePipelineLayout sum3PipelineLayout;
  vk::UniquePipeline sum3Pipeline;

  vk::UniqueBuffer materialBuffer;
  vk::UniqueDeviceMemory materialBufferMemory;

  vk::UniqueBuffer lightBuffer;
  vk::UniqueDeviceMemory lightBufferMemory;

  std::size_t countBufferSize;
  vk::UniqueBuffer countBuffer;
  vk::UniqueDeviceMemory countBufferMemory;

  std::size_t globalSize;
  vk::UniqueBuffer globalSumBuffer;
  vk::UniqueDeviceMemory globalSumBufferMemory;

  std::size_t offsetBufferSize;
  vk::UniqueBuffer offsetBuffer;
  vk::UniqueDeviceMemory offsetBufferMemory;

  std::size_t feedbackBufferSize;
  vk::UniqueBuffer feedbackBuffer;
  vk::UniqueDeviceMemory feedbackBufferMemory;

  std::size_t fragmentBufferSize;
  vk::UniqueBuffer fragmentBuffer;
  vk::UniqueDeviceMemory fragmentBufferMemory;

  std::size_t depthBufferSize;
  vk::UniqueBuffer depthBuffer;
  vk::UniqueDeviceMemory depthBufferMemory;

  std::size_t opaqueBufferSize;
  vk::UniqueBuffer opaqueBuffer;
  vk::UniqueDeviceMemory opaqueBufferMemory;

  std::size_t opaqueDepthBufferSize;
  vk::UniqueBuffer opaqueDepthBuffer;
  vk::UniqueDeviceMemory opaqueDepthBufferMemory;

  struct FrameObject {
    vk::UniqueSemaphore imageAvailableSemaphore;
    vk::UniqueSemaphore renderFinishedSemaphore;
    vk::UniqueFence inFlightFence;
    vk::UniqueFence inComputeFence;
    vk::UniqueEvent sumFinishedEvent;

    vk::UniqueCommandBuffer commandBuffer;
    vk::UniqueCommandBuffer computeCommandBuffer;

    vk::UniqueDescriptorSet descriptorSet;

    vk::UniqueBuffer uniformBuffer;
    vk::UniqueDeviceMemory uniformBufferMemory;
    void * uboData;

    vk::UniqueBuffer ssbo;
    vk::UniqueDeviceMemory ssboMemory;

    DeviceBuffer materialVertexBuffer = DeviceBuffer(vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);
    DeviceBuffer materialIndexBuffer = DeviceBuffer(vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);

    DeviceBuffer colorVertexBuffer = DeviceBuffer(vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);
    DeviceBuffer colorIndexBuffer = DeviceBuffer(vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);

    DeviceBuffer triangleVertexBuffer = DeviceBuffer(vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);
    DeviceBuffer triangleIndexBuffer = DeviceBuffer(vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);

    DeviceBuffer transparentVertexBuffer = DeviceBuffer(vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);
    DeviceBuffer transparentIndexBuffer = DeviceBuffer(vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);

    DeviceBuffer lineVertexBuffer = DeviceBuffer(vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);
    DeviceBuffer lineIndexBuffer = DeviceBuffer(vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);

    DeviceBuffer pointVertexBuffer = DeviceBuffer(vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);
    DeviceBuffer pointIndexBuffer = DeviceBuffer(vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);

    void reset() {

      // todo do something better than this

      if (*materialVertexBuffer.buffer) {

        *materialVertexBuffer.buffer = nullptr;
      }
      if (*colorVertexBuffer.buffer) {

        *colorVertexBuffer.buffer = nullptr;
      }
      if (*triangleVertexBuffer.buffer) {

        *triangleVertexBuffer.buffer = nullptr;
      }
      if (*transparentVertexBuffer.buffer) {

        *transparentVertexBuffer.buffer = nullptr;
      }
      if (*lineVertexBuffer.buffer) {

        *lineVertexBuffer.buffer = nullptr;
      }
      if (*pointVertexBuffer.buffer) {

        *pointVertexBuffer.buffer = nullptr;
      }
    }
  };

  uint32_t currentFrame = 0;
  vk::CommandBuffer currentCommandBuffer;
  std::vector<FrameObject> frameObjects;
  std::string lastAction = "";

  void setDimensions(int Width, int Height, double X, double Y);
  void updateViewmodelData();
  void setProjection();
  void update();

  static void updateHandler(int);

  static std::string getAction(int button, int mod);
  static void mouseButtonCallback(GLFWwindow * window, int button, int action, int mods);
  static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
  static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
  static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos);
  static void keyCallback(GLFWwindow * window, int key, int scancode, int action, int mods);

  void initWindow();
  void initVulkan();
  std::set<std::string> getInstanceExtensions();
  std::set<std::string> getDeviceExtensions(vk::PhysicalDevice& device);
  std::vector<const char*> getRequiredInstanceExtensions();
  void createInstance();
  void createSurface();
  void pickPhysicalDevice();
  vk::SampleCountFlagBits getMaxMSAASamples( vk::PhysicalDevice& gpu );
  QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice& physicalDevice, vk::SurfaceKHR* surface);
  bool isDeviceSuitable(vk::PhysicalDevice& device);
  bool checkDeviceExtensionSupport(vk::PhysicalDevice& device);
  void createLogicalDevice();
  SwapChainSupportDetails querySwapChainSupport(vk::PhysicalDevice device, vk::SurfaceKHR& surface);
  vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats);
  vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes);
  vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities);
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
  void createImageViews();
  void createFramebuffers();
  void createCommandPools();
  void createCommandBuffers();
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
  void recordCommandBuffer(DeviceBuffer & vertexBuffer,
                           DeviceBuffer & indexBuffer,
                           VertexBuffer * data,
                           vk::UniquePipeline & pipeline, 
                           bool incrementRenderCount=true);
  void endFrameRender();
  void endFrameCommands();
  void endFrame(int imageIndex);
  void createSyncObjects();

  uint32_t selectMemory(const vk::MemoryRequirements memRequirements, const vk::MemoryPropertyFlags properties);

  void zeroBuffer(vk::Buffer & buf, vk::DeviceSize size);
  void createBuffer(vk::Buffer& buffer, vk::DeviceMemory& bufferMemory, vk::BufferUsageFlags usage,
                    vk::MemoryPropertyFlags properties, vk::DeviceSize size);
  void createBufferUnique(vk::UniqueBuffer& buffer, vk::UniqueDeviceMemory& bufferMemory,
                          vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties,
                          vk::DeviceSize size);
  void copyBufferToBuffer(const vk::Buffer& srcBuffer, const vk::Buffer& dstBuffer, const vk::DeviceSize size);
  void copyToBuffer(const vk::Buffer& buffer, const void* data, vk::DeviceSize size,
                    vk::Buffer stagingBuffer = {}, vk::DeviceMemory stagingBufferMemory = {});
  void createImage(std::uint32_t w, std::uint32_t h, vk::SampleCountFlagBits samples, vk::Format fmt,
                   vk::ImageUsageFlags usage, vk::MemoryPropertyFlags props, vk::UniqueImage & img,
                   vk::UniqueDeviceMemory & mem);
  void createImageView(vk::Format fmt, vk::ImageAspectFlagBits flags, vk::UniqueImage& img, vk::UniqueImageView& imgView);
  void copyFromBuffer(const vk::Buffer& buffer, void* data, vk::DeviceSize size);

  void setDeviceBufferData(DeviceBuffer& buffer, const void* data, vk::DeviceSize size, std::size_t nobjects=0);

  void createDescriptorSetLayout();
  void createComputeDescriptorSetLayout();
  // void createUniformBuffers();
  void createDescriptorPool();
  void createComputeDescriptorPool();
  void createDescriptorSets();
  void writeDescriptorSets();
  void updateSceneDependentBuffers();

  void createMaterialVertexBuffer();
  void createMaterialIndexBuffer();

  void createBuffers();
  void createDependentBuffers();

  void createCountRenderPass();
  void createGraphicsRenderPass();
  void createOpaqueRenderPass();
  void createTransparentRenderPass();
  void createBlendRenderPass();
  void createGraphicsPipelineLayout();
  template<typename V>
  void createGraphicsPipeline(PipelineType type, vk::UniquePipeline & graphicsPipeline, vk::PrimitiveTopology topology,
                              vk::PolygonMode fillMode, std::string const & shader,
                              int graphicsSubpass, bool enableDepthWrite=true,
                              bool transparent=false, bool disableMultisample=false);
  void createGraphicsPipelines();
  void createComputePipeline(vk::UniquePipelineLayout & layout, vk::UniquePipeline & pipeline,
                             std::string const & shaderFile);
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
  void partialSums(bool readSize=false);
  void resizeBlendShader(std::uint32_t maxDepth);
  void resizeFragmentBuffer(FrameObject & object);
  void refreshBuffers(FrameObject & object, int imageIndex);
  void blendFrame(int imageIndex);
  void drawBuffers(FrameObject & object, int imageIndex);
  void drawFrame();
  void recreateSwapChain();
  vk::UniqueShaderModule createShaderModule(const std::vector<char>& code);
  void nextFrame();
  void display();
  void poll();
  void mainLoop();
  void cleanup();

  void idleFunc(std::function<void()> f);
  void idle();

  // user controls
  void Export(int imageIndex);
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

  void shift(double dx, double dy);
  void pan(double dx, double dy);
  void capzoom();
  void zoom(double dx, double dy);
  void travelHome();
  void cycleMode();
};

extern AsyVkRender* vk;

} // namespace camp
