/*****
 * glrender.h
 * Render 3D Bezier paths and surfaces.
 *****/

#ifndef GLRENDER_H
#define GLRENDER_H

#include "common.h"
#include "glmCommon.h"
#include "triple.h"
#include "pen.h"

#ifdef HAVE_RENDERER

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

#ifdef HAVE_LIBOSMESA
#ifndef APIENTRY
#define APIENTRY
#endif
#ifndef GLAPI
#define GLAPI
#endif
#define GLEW_OSMESA
#include <GL/osmesa.h>
#endif

#ifdef HAVE_LIBGLFW
#include <GLFW/glfw3.h>
#include "glfw.h"
#endif

#else
typedef unsigned int GLuint;
typedef int GLint;
typedef float GLfloat;
typedef double GLdouble;
typedef unsigned char GLubyte;
typedef unsigned int GLenum;
#define GL_POINTS				0x0000
#define GL_LINES				0x0001
#define GL_TRIANGLES				0x0004
#endif

#ifdef HAVE_LIBGLM
#include "material.h"
#endif

#include "renderBase.h"
#include "render.h"

namespace camp {
class picture;
}

namespace camp {

// Forward declarations for texture types (defined in GLTextures.h)
#ifdef HAVE_LIBGLM
template<typename T, GLuint GLDataType> class GLTexture2;
template<typename T, GLuint GLDataType> class GLTexture3;
#endif

// Global BBT matrix for billboard transformations (accessed from multiple translation units)
extern double BBT[9];

// Accessor functions for matrices (to avoid synchronization with gl instance)
#ifdef HAVE_LIBGLM
const glm::dmat4& getProjViewMat();
const glm::dmat4& getViewMat();
const glm::dmat3& getNormMat();
#endif

// Projection matrix pointer for shader compatibility (following Vulkan pattern)
#ifdef HAVE_LIBGLM
extern const double* dprojView;  // For drawelement.h Transform2T
#endif

struct Billboard {
  double cx,cy,cz;

  void init(const triple& center) {
    cx=center.getx();
    cy=center.gety();
    cz=center.getz();
  }

  triple transform(const triple& v) const {
    return triple((v.getx()-cx)*BBT[0]+(v.gety()-cy)*BBT[3]+(v.getz()-cz)*BBT[6]+cx,
                  (v.getx()-cx)*BBT[1]+(v.gety()-cy)*BBT[4]+(v.getz()-cz)*BBT[7]+cy,
                  (v.getx()-cx)*BBT[2]+(v.gety()-cy)*BBT[5]+(v.getz()-cz)*BBT[8]+cz);
  }
};

extern Billboard BB;
#ifdef HAVE_RENDERER
extern GLuint vao;  // Vertex Array Object
#endif

#ifdef HAVE_LIBGLM
extern std::vector<Material> materials;
extern MaterialMap materialMap;
extern size_t materialIndex;
extern int MaterialIndex;

extern const size_t Nbuffer; // Initial size of 2D dynamic buffers
extern const size_t nbuffer; // Initial size of 0D & 1D dynamic buffers

// VertexBuffer and related types are defined in render.h
// Globals: materialData, colorData, triangleData, transparentData, pointData, lineData

void drawBuffer(VertexBuffer& data, GLint shader, bool color=false, unsigned int drawType=4);  // drawType: 0=GL_POINTS, 1=GL_LINES, 4=GL_TRIANGLES
void drawBuffers();

void clearMaterials();
void clearCenters();

void drawMaterial0();
void drawMaterial1();
void drawMaterial();
void drawColor();
void drawTriangle();
void drawTransparent();

#endif

#ifdef HAVE_RENDERER
// OpenGL renderer class following Vulkan pattern
class AsyGLRender : public AsyRender, public RenderCallbacks
{
public:
  AsyGLRender() = default;
  ~AsyGLRender();

  void updateModelViewData() override;

  void render(RenderFunctionArgs const& args) override;

  // RenderCallbacks interface implementation (GLFW callbacks)
  void onMouseButton(int button, int action, int mods) override;
  void onFramebufferResize(int width, int height) override;
  void onScroll(double xoffset, double yoffset) override;
  void onCursorPos(double xpos, double ypos) override;
  void onKey(int key, int scancode, int action, int mods) override;
  void onWindowFocus(int focused) override;
  void onClose() override;

  // OpenGL-specific state (mirroring AsyVkRender pattern)
  bool outlinemode = false;
  bool glupdate = false;
  bool glexit = false;
  bool initialized = false;
  bool copied = false;

  // Lighting (OpenGL-specific, public for jsfile/v3dfile access)
  size_t Nlights = 1;
  size_t nlights0 = 0;  // Saved original number of lights for mode restoration
  camp::triple* Lights = nullptr;
  double* Diffuse = nullptr;
  double* Specular = nullptr;

#ifdef HAVE_PTHREAD
  pthread_cond_t initSignal = PTHREAD_COND_INITIALIZER;
  pthread_mutex_t initLock = PTHREAD_MUTEX_INITIALIZER;
  pthread_cond_t readySignal = PTHREAD_COND_INITIALIZER;
  pthread_mutex_t readyLock = PTHREAD_MUTEX_INITIALIZER;
#endif

public:
  // OpenGL-specific members (following Vulkan pattern)
#ifdef HAVE_RENDERER
  // Shaders - made public for standalone function access during refactoring
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
  GLint sum2fastShader = 0;
  GLint sum3Shader = 0;

  // VAO and buffers - made public for standalone function access during refactoring
  GLuint vao = 0;
  GLuint materialsBuffer = 0;  // Uniform buffer for materials
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
  GLuint lastpixels = 0;
  int maxTileWidth = 1024;
  int maxTileHeight = 768;

  // Rendering state
  GLint lastshader = -1;
  bool ssbo = false;
  bool interlock = false;
  bool initSSBO = true;
  GLuint fragments = 0;
  GLuint maxFragments = 0;
  GLuint maxSize = 1;

  // GPU settings
  GLuint g = 0;
  GLuint processors = 0;
  GLuint localSize = 0;
  GLuint blockSize = 0;
  GLuint groupSize = 0;

  // IBL textures - kept as globals in glrender.cc where GLTextures.h is available

  // Mouse interaction state
  double xprev = 0.0;
  double yprev = 0.0;
  double lastangle = 0.0;
  string currentAction = "";

  // Window state
  bool queueExport = false;
  bool readyAfterExport = false;
  bool havewindow = false;
  bool format3dWait = false;
  bool exporting = false;
  int oldWidth = 0;
  int oldHeight = 0;
  bool firstFit = true;

  // Spin state
  bool Xspin = false;
  bool Yspin = false;
  bool Zspin = false;

  // Timer for FPS measurement
  utils::stopWatch spinTimer;
#endif

public:
  void update();
  void cycleMode() override;

  /** Returns the GLFW window pointer (does the static_cast from void* once) */
  GLFWwindow* getGLFWWindow() const { return static_cast<GLFWwindow*>(glfwWindow); }

protected:
  void mainLoop();
  void display();
  void exportHandler(int = 0);
  void updateHandler(int = 0);
  virtual void reshape0(int width, int height) override;
};

// Global OpenGL renderer instance (defined in picture.cc)
extern AsyGLRender* gl;

void frustum(double left, double right, double bottom,
             double top, double nearVal, double farVal);
void ortho(double left, double right, double bottom,
           double top, double nearVal, double farVal);

#endif // HAVE_RENDERER

} // namespace camp

#endif  // GLRENDER_H
