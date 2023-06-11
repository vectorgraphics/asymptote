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

typedef mem::map<const Material, size_t> MaterialMap;

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

// struct MaterialVertex {
//   glm::vec3 position;
//   glm::vec3 normal;
//   glm::i32 material; // should this be a signed or unsigned int?

//   static vk::VertexInputBindingDescription getBindingDescription()
//   {
//     return vk::VertexInputBindingDescription(0, sizeof(MaterialVertex), vk::VertexInputRate::eVertex);
//   }

//   static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescriptions()
//   {
//     return std::array<vk::VertexInputAttributeDescription, 3>{
//             vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, offsetof(MaterialVertex, position)),
//             vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat, offsetof(MaterialVertex, normal)),
//             vk::VertexInputAttributeDescription(2, 0, vk::Format::eR32Sint, offsetof(MaterialVertex, material))};
//   }
// };

// struct ColorVertex {
//   glm::vec3 position;
//   glm::vec3 normal;
//   glm::i32 material;
//   glm::vec3 color;

//   static vk::VertexInputBindingDescription getBindingDescription()
//   {
//     return vk::VertexInputBindingDescription(0, sizeof(ColorVertex), vk::VertexInputRate::eVertex);
//   }

//   static std::array<vk::VertexInputAttributeDescription, 4> getAttributeDescriptions()
//   {
//     return std::array<vk::VertexInputAttributeDescription, 4>{
//             vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, offsetof(ColorVertex, position)),
//             vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat, offsetof(ColorVertex, normal)),
//             vk::VertexInputAttributeDescription(2, 0, vk::Format::eR32Sint, offsetof(ColorVertex, material)),
//             vk::VertexInputAttributeDescription(3, 0, vk::Format::eR32G32B32Sfloat, offsetof(ColorVertex, color))};
//   }
// };

// struct TriangleVertex {
//   glm::vec3 position;
//   glm::vec3 normal;
//   glm::i32 material;
//   glm::vec3 color;

//   static vk::VertexInputBindingDescription getBindingDescription()
//   {
//     return vk::VertexInputBindingDescription(0, sizeof(TriangleVertex), vk::VertexInputRate::eVertex);
//   }

//   static std::array<vk::VertexInputAttributeDescription, 4> getAttributeDescriptions()
//   {
//     return std::array<vk::VertexInputAttributeDescription, 4>{
//             vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, offsetof(TriangleVertex, position)),
//             vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat, offsetof(TriangleVertex, normal)),
//             vk::VertexInputAttributeDescription(2, 0, vk::Format::eR32Sint, offsetof(TriangleVertex, material)),
//             vk::VertexInputAttributeDescription(3, 0, vk::Format::eR32G32B32Sfloat, offsetof(TriangleVertex, color))};
//   }
// };

// struct TransparentVertex {
//   glm::vec3 position;
//   glm::vec3 normal;
//   glm::i32 material;
//   glm::vec4 color;

//   static vk::VertexInputBindingDescription getBindingDescription()
//   {
//     return vk::VertexInputBindingDescription(0, sizeof(TransparentVertex), vk::VertexInputRate::eVertex);
//   }

//   static std::array<vk::VertexInputAttributeDescription, 4> getAttributeDescriptions()
//   {
//     return std::array<vk::VertexInputAttributeDescription, 4>{
//             vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, offsetof(TransparentVertex, position)),
//             vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat, offsetof(TransparentVertex, normal)),
//             vk::VertexInputAttributeDescription(2, 0, vk::Format::eR32Sint, offsetof(TransparentVertex, material)),
//             vk::VertexInputAttributeDescription(3, 0, vk::Format::eR32G32B32Sfloat, offsetof(TransparentVertex, color))};
//   }
// };

// struct LineVertex {
//   glm::vec3 position;
//   glm::vec3 normal;
//   glm::i32 material;
//   // glm::vec3 color;

//   static vk::VertexInputBindingDescription getBindingDescription()
//   {
//     return vk::VertexInputBindingDescription(0, sizeof(LineVertex), vk::VertexInputRate::eVertex);
//   }

//   static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescriptions()
//   {
//     return std::array<vk::VertexInputAttributeDescription, 3>{
//             vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, offsetof(LineVertex, position)),
//             vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat, offsetof(LineVertex, normal)),
//             vk::VertexInputAttributeDescription(2, 0, vk::Format::eR32Sint, offsetof(LineVertex, material))};
//   }
// };

// struct PointVertex {
//   glm::vec3 position;
//   glm::i32 material;
//   glm::f32 width;
//   // glm::vec3 color;

//   static vk::VertexInputBindingDescription getBindingDescription()
//   {
//     return vk::VertexInputBindingDescription(0, sizeof(PointVertex), vk::VertexInputRate::eVertex);
//   }

//   static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescriptions()
//   {
//     return std::array<vk::VertexInputAttributeDescription, 3>{
//             vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, offsetof(PointVertex, position)),
//             vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32Sint, offsetof(PointVertex, material)),
//             vk::VertexInputAttributeDescription(2, 0, vk::Format::eR32Sfloat, offsetof(PointVertex, width))};
//   }
// };

// template<class T>
// struct VertexQueue {
//   std::vector<T> vertices;
//   std::vector<glm::u32> indices;

//   // std::vector<Material> materials;

//   void extend(const VertexQueue<T>& other)
//   {
//     extendOffset<glm::u32>(indices, other.indices, vertices.size());
//     vertices.insert(vertices.end(), other.vertices.begin(), other.vertices.end());
//     // materials.insert(materials.end(), other.materials.begin(), other.materials.end());
//   }

//   // combine this? addIndexedVertex?
//   size_t addVertex(const T& vertex)
//   {
//     size_t nvertices = vertices.size();
//     vertices.push_back(vertex);
//     return nvertices;
//   }
// };

struct MaterialVertex // vertexData
{
  glm::vec3 position;
  glm::vec3 normal;
  glm::i32 material;
  glm::vec4 color;

  static vk::VertexInputBindingDescription getBindingDescription()
  {
    return vk::VertexInputBindingDescription(0, sizeof(MaterialVertex), vk::VertexInputRate::eVertex);
  }

  static std::array<vk::VertexInputAttributeDescription, 4> getAttributeDescriptions()
  {
    return std::array<vk::VertexInputAttributeDescription, 4>{
            vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, offsetof(MaterialVertex, position)),
            vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat, offsetof(MaterialVertex, normal)),
            vk::VertexInputAttributeDescription(2, 0, vk::Format::eR32Sint, offsetof(MaterialVertex, material)),
            vk::VertexInputAttributeDescription(3, 0, vk::Format::eR32G32B32Sfloat, offsetof(MaterialVertex, color))};
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
            vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, offsetof(ColorVertex, position)),
            vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat, offsetof(ColorVertex, normal)),
            vk::VertexInputAttributeDescription(2, 0, vk::Format::eR32Sint, offsetof(ColorVertex, material)),
            vk::VertexInputAttributeDescription(3, 0, vk::Format::eR32G32B32Sfloat, offsetof(ColorVertex, color))};
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
            vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, offsetof(PointVertex, position)),
            vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32Sfloat, offsetof(PointVertex, width)),
            vk::VertexInputAttributeDescription(2, 0, vk::Format::eR32Sint, offsetof(PointVertex, material))};
  }
};

struct VertexBuffer {
  std::vector<MaterialVertex> materialVertices;
  std::vector<ColorVertex> colorVertices;
  std::vector<PointVertex> pointVertices;
  std::vector<GLuint> indices;

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

enum FlagsPushConstant: unsigned int
{
  PUSHFLAGS_NONE    = 0,
  PUSHFLAGS_NOLIGHT = 1 << 0,
  PUSHFLAGS_COLORED = 1 << 1,
  PUSHFLAGS_GENERAL = 1 << 2
};

struct PushConstants
{
  glm::uvec4 constants;
  // constants[0] = flags
  // constants[1] = nlights
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
    bool display;
    std::string title;
    int maxFramesInFlight = 2;
    vk::PresentModeKHR presentMode = vk::PresentModeKHR::eImmediate; //vk::PresentModeKHR::eFifo;
    vk::SampleCountFlagBits samples = vk::SampleCountFlagBits::e1;
  };

  Options options;

  AsyVkRender(Options& options);
  ~AsyVkRender();

  void vkrender(const picture* pic, const string& format,
                double width, double height, double angle, double zoom,
                const triple& m, const triple& M, const pair& shift,
                const pair& margin, double* t,
                double* background, size_t nlightsin, triple* lights,
                double* diffuse, double* specular, bool view);

  triple billboardTransform(const triple& center, const triple& v) const;
  double getRenderResolution(triple Min) const;

  bool framebufferResized = false;
  bool recreatePipeline = false;
  bool newBufferData = true;
  bool newUniformBuffer = true;

  // VertexQueue<MaterialVertex> materialVertices;
  // VertexQueue<ColorVertex> colorVertices;
  // VertexQueue<TriangleVertex> triangleVertices;
  // VertexQueue<TransparentVertex> transparentVertices;
  // VertexQueue<LineVertex> lineVertices;
  // VertexQueue<PointVertex> pointVertices;

  VertexBuffer materialData;
  VertexBuffer colorData;
  VertexBuffer triangleData;
  VertexBuffer transparentData;
  VertexBuffer lineData;
  VertexBuffer pointData;

  // clear every frame?
  std::vector<Material> materials;
  MaterialMap materialMap;
  size_t materialIndex;

  bool outlinemode = false;

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

  int fullWidth, fullHeight; // TODO: pixel density, expand?
  // What is the difference between these?
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

  size_t Nmaterials;   // Number of materials compiled in shader
  size_t nmaterials;   // Current size of materials buffer
  size_t Maxmaterials; // Maxinum size of materials buffer


  void updateProjection();
  void frustum(GLdouble left, GLdouble right, GLdouble bottom,
               GLdouble top, GLdouble nearVal, GLdouble farVal);
  void ortho(GLdouble left, GLdouble right, GLdouble bottom,
             GLdouble top, GLdouble nearVal, GLdouble farVal);

  void clearVertexBuffers();
  void clearCenters();
  void clearMaterials();

private:
  struct DeviceBuffer {
    vk::BufferUsageFlags usage;
    vk::MemoryPropertyFlags properties;
    vk::DeviceSize bufferSize = 0;
    vk::DeviceSize memorySize = 0;
    vk::UniqueBuffer buffer;
    vk::UniqueDeviceMemory memory;
    // only used if hasExternalMemoryHostExtension == false
    vk::UniqueBuffer stagingBuffer;
    vk::UniqueDeviceMemory stagingBufferMemory;

    DeviceBuffer(vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties)
        : usage(usage), properties(properties) {}
  };


  const picture* pic;

  double H;
  double xfactor, yfactor; // what is this for?

  double x, y; // make these more descriptive (something to do with shift?)

  double cx, cy; // center variables

  int screenWidth, screenHeight;
  int width, height; // width and height of the window
  double aspect;
  double oWidth, oHeight;
  double lastZoom;

  utils::stopWatch spinTimer;
  utils::stopWatch fpsTimer;
  utils::statistics fpsStats;
  std::function<void()> currentIdleFunc = nullptr;
  bool Xspin = false;
  bool Yspin = false;
  bool Zspin = false;
  bool Animate = false;
  bool Step = false;

  bool remesh = true; // whether picture needs to be remeshed
  bool redraw = true;  // whether a new frame needs to be rendered

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
  std::vector<vk::Image> swapChainImages;
  vk::Format swapChainImageFormat;
  vk::Extent2D swapChainExtent;
  std::vector<vk::UniqueImageView> swapChainImageViews;
  std::vector<vk::UniqueFramebuffer> swapChainFramebuffers;

  vk::UniqueImage depthImage;
  vk::UniqueImageView depthImageView;
  vk::UniqueDeviceMemory depthImageMemory;

  vk::SampleCountFlagBits msaaSamples;
  vk::UniqueImage colorImage;
  vk::UniqueImageView colorImageView;
  vk::UniqueDeviceMemory colorImageMemory;

  vk::UniqueCommandPool transferCommandPool;
  vk::UniqueCommandPool renderCommandPool;

  vk::UniqueDescriptorPool descriptorPool;

  vk::UniqueRenderPass materialRenderPass;
  vk::UniqueDescriptorSetLayout materialDescriptorSetLayout;

  vk::UniquePipelineLayout materialPipelineLayout;
  vk::UniquePipeline materialPipeline;

  vk::UniquePipelineLayout linePipelineLayout;
  vk::UniquePipeline linePipeline;

  vk::UniquePipelineLayout pointPipelineLayout;
  vk::UniquePipeline pointPipeline;

  vk::UniqueDescriptorSetLayout computeDescriptorSetLayout;
  vk::UniquePipelineLayout computePipelineLayout;
  vk::UniquePipeline computePipeline;

  vk::UniqueBuffer materialBuffer;
  vk::UniqueDeviceMemory materialBufferMemory;

  vk::UniqueBuffer lightBuffer;
  vk::UniqueDeviceMemory lightBufferMemory;

  struct FrameObject {
    vk::UniqueSemaphore imageAvailableSemaphore;
    vk::UniqueSemaphore renderFinishedSemaphore;
    vk::UniqueFence inFlightFence;

    vk::UniqueCommandBuffer commandBuffer;

    vk::UniqueDescriptorSet descriptorSet;

    vk::UniqueBuffer uniformBuffer;
    vk::UniqueDeviceMemory uniformBufferMemory;

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
  };

  uint32_t currentFrame = 0;
  std::vector<FrameObject> frameObjects;
  std::string lastAction = "";

  void setDimensions(int Width, int Height, double X, double Y);
  void updateViewmodelData();
  void setProjection();
  void update();

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
  void createSwapChain();
  void createImageViews();
  void createFramebuffers();
  void createCommandPools();
  void createCommandBuffers();
  PushConstants buildPushConstants(FlagsPushConstant addFlags);
  vk::CommandBuffer & getCommandBuffer();
  void beginFrame(uint32_t imageIndex);
  void recordCommandBuffer(DeviceBuffer & vertexBuffer, DeviceBuffer & indexBuffer, VertexBuffer * data, vk::UniquePipeline & pipeline, FlagsPushConstant addFlags = PUSHFLAGS_NONE);
  void endFrame();
  void createSyncObjects();

  uint32_t selectMemory(const vk::MemoryRequirements memRequirements, const vk::MemoryPropertyFlags properties);

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
  // void copyFromBuffer(const vk::Buffer& buffer, void* data, vk::DeviceSize size,
  //                     bool wait = true, vk::Fence fence = {}, const vk::Semaphore semaphore = {},
  //                     vk::Buffer stagingBuffer = {}, vk::DeviceMemory stagingBufferMemory = {});

  void setDeviceBufferData(DeviceBuffer& buffer, const void* data, vk::DeviceSize size);

  void createDescriptorSetLayout();
  void createComputeDescriptorSetLayout();
  // void createUniformBuffers();
  void createDescriptorPool();
  void createDescriptorSets();

  void createMaterialVertexBuffer();
  void createMaterialIndexBuffer();

  void createBuffers();

  void createMaterialRenderPass();
  template<typename V>
  void createGraphicsPipeline(vk::UniquePipelineLayout & layout, vk::UniquePipeline & pipeline,
                              vk::PrimitiveTopology topology, vk::PolygonMode fillMode,
                              std::string const & shaderFile);
  void createGraphicsPipelines();
  void createComputePipeline();

  void createAttachments();

  void updateUniformBuffer(uint32_t currentImage);
  void updateBuffers();
  void drawPoints(FrameObject & object);
  void drawLines(FrameObject & object);
  void drawMaterials(FrameObject & object);
  void drawColors(FrameObject & object);
  void drawTriangles(FrameObject & object);
  void drawFrame();
  void recreateSwapChain();
  vk::UniqueShaderModule createShaderModule(const std::vector<char>& code);
  void display();
  void mainLoop();
  void cleanup();

  void idleFunc(std::function<void()> f);
  void idle();

  // user controls
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
