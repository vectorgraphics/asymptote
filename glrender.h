/*****
 * glrender.h
 * Render 3D Bezier paths and surfaces.
 *****/

#ifndef GLRENDER_H
#define GLRENDER_H

#include "common.h"
#include "renderBase.h"
#include "render.h"

#ifdef HAVE_GL

#include <unordered_map>
#include <csignal>

#ifdef __APPLE__
#define GL_SILENCE_DEPRECATION
#endif
#include "GL/glew.h"

#ifdef __APPLE__
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

#include "glmCommon.h"
#include "triple.h"
#include "pen.h"

#include <GLFW/glfw3.h>
#include "glfw.h"

#include "material.h"

namespace camp {

class picture;

// Forward declarations for texture types (defined in GLTextures.h)
template<typename T, GLuint GLDataType> class GLTexture2;
template<typename T, GLuint GLDataType> class GLTexture3;

// Projection matrix pointer for shader compatibility (following Vulkan pattern)
extern const double* dprojView;  // For drawelement.h Transform2T

extern GLuint vao;  // Vertex Array Object

extern size_t materialIndex;
extern int MaterialIndex;

// VertexBuffer and related types are defined in render.h
// Globals: materialData, colorData, triangleData, transparentData, pointData, lineData

void clearMaterials();
void clearCenters();

// OpenGL renderer class following Vulkan pattern
class AsyGLRender : public AsyRender, public RenderCallbacks
{
public:
  AsyGLRender() = default;
  ~AsyGLRender();

  void render(RenderFunctionArgs const& args) override;

  // RenderCallbacks interface implementation (GLFW callbacks)
  void onScroll(double xoffset, double yoffset) override;
  void onMouseButton(int button, int action, int mods) override;
  void onFramebufferResize(int width, int height) override;
  void onCursorPos(double xpos, double ypos) override;
  void onKey(int key, int scancode, int action, int mods) override;
  void onWindowFocus(int focused) override;
  void onClose() override;

  // Virtual overrides called by base class
  void update() override;
  void cycleMode() override;
  void drawFrame() override;
  void swapBuffers() override;
  void Export(int imageIndex=0) override;

  GLFWwindow* getRenderWindow() const;

private:
  bool GPUindexing=false;
  bool GPUcompress;
  bool shouldUpdateBuffers = true;
  bool copied = false;
  size_t Nlights = 1;
  size_t nmaterials = 0;
  size_t nlights0 = 0;

  // Shaders
  GLint pixelShader = 0;
  GLint materialShader[2] = {0, 0};
  GLint colorShader[2] = {0, 0};
  GLint generalShader[2] = {0, 0};
  GLint countShader = 0;
  GLint transparentShader = 0;
  GLint blendShader = 0;
  GLint zeroShader = 0;
  GLint compressShader = 0;
  GLint sum1Shader = 0;
  GLint sum2Shader = 0;
  GLint sum3Shader = 0;

  // VAO and buffers
  GLuint vao = 0;
  GLuint materialsBuffer = 0;
  GLuint offsetBuffer = 0;
  GLuint indexBuffer = 0;
  GLuint elementsBuffer = 0;
  GLuint countBuffer = 0;
  GLuint globalSumBuffer = 0;
  GLuint fragmentBuffer = 0;
  GLuint depthBuffer = 0;
  GLuint opaqueBuffer = 0;
  GLuint opaqueDepthBuffer = 0;
  GLuint feedbackBuffer = 0;

  // Framebuffers/textures (for export)
  GLuint pixels = 0;
  GLuint elements = 0;

  // Rendering state
  GLint lastshader = -1;
  GLint projViewLoc = -1;
  GLint viewMatLoc = -1;
  GLint normMatLoc = -1;

  // Persistent GL buffer handles per VertexBuffer instance
  struct GLBufferPair { GLuint vertexBuffer=0; GLuint indexBuffer=0; };
  std::unordered_map<VertexBuffer*, GLBufferPair> glBuffers;
  GLuint fragments = 0;
  GLuint maxFragments = 0;
  GLuint maxSize = 1;

  // GPU settings
  GLuint g = 0;
  GLuint processors = 0;
  GLuint localSize = 0;
  GLuint blockSize = 0;
  GLuint groupSize = 0;

  // Interaction state
  string currentAction = "";
  bool exporting = false;
  utils::stopWatch spinTimer;

  // Shader and buffer management
  void initComputeShaders();
  void initBlendShader();
  void setBuffers();
  void initShaders();
  void deleteComputeShaders();
  void deleteBlendShader();
  void deleteShaders();
  void resizeBlendShader(GLuint maxDepth);

  // Rendering functions
  void refreshBuffers();
  void setUniformsOpenGL(GLint shader);
  void drawBuffer(VertexBuffer& data, GLint shader, bool color=false, unsigned int drawType=4);
  void drawPoints();
  void drawLines();
  void drawMaterials();
  void drawColors();
  void drawTriangles();
  void aBufferTransparency();
  void drawTransparent();
  void drawBuffers();

  // GPU compute functions
  void clearCount();
  void compressCount();
  void partialSums(bool readSize=false);
  void resizeFragmentBuffer();

protected:
  void exportHandler(int) override;
  virtual void reshape(int width, int height) override;
};

// Note: The global renderer pointer 'gl' is declared in renderBase.h as AsyRender*
// This allows unified access to both OpenGL and Vulkan renderers through the base class

void frustum(double left, double right, double bottom,
             double top, double nearVal, double farVal);
void ortho(double left, double right, double bottom,
           double top, double nearVal, double farVal);

// No-SSBO fallback: sort transparent triangles by centroid depth
void sortTriangles();

} // namespace camp

#endif // HAVE_GL

#endif  // GLRENDER_H
