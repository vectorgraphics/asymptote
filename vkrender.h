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
#ifdef HAVE_RENDERER
#include <vma_cxx.h>

#include <glslang/Public/ShaderLang.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#endif

#include "material.h"
#include "pen.h"
#include "triple.h"
#include "seconds.h"
#include "statistics.h"

#include "render.h"
#include "renderBase.h"
#include "glfw.h"

namespace camp
{
class picture;

std::vector<char> readFile(const std::string& filename);

#ifdef HAVE_RENDERER
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
  glm::mat4 projViewMat;
  glm::mat4 viewMat;
  // GLSL mat3 in std140 = 3 columns of vec4 (48 bytes)
  glm::vec4 normMat[3];
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

extern glm::dmat4 projViewMat;  // Deprecated - use getProjViewMat() instead
extern glm::dmat4 normMat;      // Deprecated - use getNormMat() instead

// Accessor functions to avoid synchronization with vk instance
const glm::dmat4& getProjViewMat();
const glm::dmat3& getNormMat();

class AsyVkRender : public AsyRender, public RenderCallbacks
{
public:

  AsyVkRender() = default;
  ~AsyVkRender();

  // Implementation of base class pure virtual (actual rendering implementation)
  void render(RenderFunctionArgs const& args) override;

  // RenderCallbacks interface implementation (GLFW callbacks)
  void onMouseButton(int button, int action, int mods) override;
  void onFramebufferResize(int width, int height) override;
  void onScroll(double xoffset, double yoffset) override;
  void onCursorPos(double xpos, double ypos) override;
  void onKey(int key, int scancode, int action, int mods) override;
  void onWindowFocus(int focused) override;
  void onClose() override;

  bool framebufferResized=false;
  bool recreatePipeline=false;
  bool recreateBlendPipeline=false;
  bool shouldUpdateBuffers=true;
  bool newUniformBuffer=true;
  // Note: ibl is now in base class AsyRender
  bool vkexit=false;

  int maxFramesInFlight;

#ifdef HAVE_RENDERER
  vk::SampleCountFlagBits samples = vk::SampleCountFlagBits::e1;
#endif

  std::uint32_t pixels;

  const double* dprojView;
  const double* dView;
private:
#ifdef HAVE_RENDERER
  static constexpr std::array<const char*, 4> deviceExtensions = {
    VK_KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME,
    VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME,
    VK_KHR_MULTIVIEW_EXTENSION_NAME,
    VK_KHR_MAINTENANCE2_EXTENSION_NAME
  };

  static constexpr auto VB_USAGE_FLAGS = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer;
  static constexpr auto IB_USAGE_FLAGS = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer;

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

#ifdef HAVE_RENDERER

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
#endif

protected:
  void updateModelViewData() override;
  void setProjection() override;

public:
  void updateHandler(int=0) override;

#ifdef HAVE_RENDERER
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
                  vk::Pipeline pipeline);
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
  void createComputePipelineOnly(
    vk::PipelineLayout layout,
    vk::UniquePipeline & pipeline,
    std::string const& shaderFile
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
  void drawFrame() override;
  void swapBuffers() override;
  void showWindow() override;
  void recreateSwapChain();
  void initializeSwapChainIfNeeded();
  vk::UniqueShaderModule createShaderModule(EShLanguage lang, std::string const & filename, std::vector<std::string> const & options);
#endif

  GLFWwindow* getRenderWindow() const;
  void cleanup();

  // user controls
  void exportHandler(int=0) override;
  void Export(int imageIndex=0) override;
  bool readyForUpdate=false;
  // Note: initialized and format3dWait are now in base class AsyRender

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

  // Graphics library cleanup
  void finalizeProcess() override;

  /** Returns the GLFW window pointer (does the static_cast from void* once) */
  GLFWwindow* getGLFWWindow() const { return static_cast<GLFWwindow*>(glfwWindow); }

  // Vulkan-specific overrides that add to base class behavior
  virtual void reshape(int width, int height) override;
  virtual void cycleMode() override;

  friend void glfwInitWindow(AsyRender*, int, int, const std::string&);
  friend void glfwCleanupWindow(AsyVkRender*);
};

} // namespace camp
