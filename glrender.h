/*****
 * glrender.h
 * Render 3D Bezier paths and surfaces.
 *****/

#ifndef GLRENDER_H
#define GLRENDER_H

#include "common.h"
#include "triple.h"
#include "pen.h"

#ifdef HAVE_LIBGLM
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/transform.hpp>
#endif

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
extern double glBBT[9];

// Projection matrices for shader compatibility (following Vulkan pattern)
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
    return triple((v.getx()-cx)*glBBT[0]+(v.gety()-cy)*glBBT[3]+(v.getz()-cz)*glBBT[6]+cx,
                  (v.getx()-cx)*glBBT[1]+(v.gety()-cy)*glBBT[4]+(v.getz()-cz)*glBBT[7]+cy,
                  (v.getx()-cx)*glBBT[2]+(v.gety()-cy)*glBBT[5]+(v.getz()-cz)*glBBT[8]+cz);
  }
};

extern Billboard BB;
#ifdef HAVE_RENDERER
extern GLuint vao;  // Vertex Array Object
#endif

#ifdef HAVE_LIBGLM
typedef mem::map<const Material,size_t> MaterialMap;

extern std::vector<Material> materials;
extern MaterialMap materialMap;
extern size_t materialIndex;
extern int MaterialIndex;

extern const size_t Nbuffer; // Initial size of 2D dynamic buffers
extern const size_t nbuffer; // Initial size of 0D & 1D dynamic buffers

class vertexData
{
public:
  GLfloat position[3];
  GLfloat normal[3];
  GLint material;
  vertexData() {};
  vertexData(const triple& v, const triple& n) {
    position[0]=v.getx();
    position[1]=v.gety();
    position[2]=v.getz();
    normal[0]=n.getx();
    normal[1]=n.gety();
    normal[2]=n.getz();
    material=MaterialIndex;
  }
};

class VertexData
{
public:
  GLfloat position[3];
  GLfloat normal[3];
  GLint material;
  GLfloat color[4];
  VertexData() {};
  VertexData(const triple& v, const triple& n) {
    position[0]=v.getx();
    position[1]=v.gety();
    position[2]=v.getz();
    normal[0]=n.getx();
    normal[1]=n.gety();
    normal[2]=n.getz();
    material=MaterialIndex;
  }
  VertexData(const triple& v, const triple& n, GLfloat *c) {
    position[0]=v.getx();
    position[1]=v.gety();
    position[2]=v.getz();
    normal[0]=n.getx();
    normal[1]=n.gety();
    normal[2]=n.getz();
    material=MaterialIndex;
    color[0]=c[0];
    color[1]=c[1];
    color[2]=c[2];
    color[3]=c[3];
  }
};

class vertexData0 {
public:
  GLfloat position[3];
  GLfloat width;
  GLint material;
  vertexData0() {};
  vertexData0(const triple& v, double width) : width(width) {
    position[0]=v.getx();
    position[1]=v.gety();
    position[2]=v.getz();
    material=MaterialIndex;
  }
};

class vertexBuffer {
public:
  GLenum type;

  GLuint verticesBuffer;
  GLuint VerticesBuffer;
  GLuint vertices0Buffer;
  GLuint indicesBuffer;
  GLuint materialsBuffer;

  std::vector<vertexData> vertices;
  std::vector<VertexData> Vertices;
  std::vector<vertexData0> vertices0;
  std::vector<GLuint> indices;

  std::vector<Material> materials;
  std::vector<GLint> materialTable;

  bool rendered; // Are all patches in this buffer fully rendered?
  bool partial;  // Does buffer contain incomplete data?

  vertexBuffer(GLint type=GL_TRIANGLES) : type(type),
                                          verticesBuffer(0),
                                          VerticesBuffer(0),
                                          vertices0Buffer(0),
                                          indicesBuffer(0),
                                          materialsBuffer(0),
                                          rendered(false),
                                          partial(false)
  {}

  void clear() {
    vertices.clear();
    Vertices.clear();
    vertices0.clear();
    indices.clear();
    materials.clear();
    materialTable.clear();
  }

  void reserve0() {
    vertices0.reserve(nbuffer);
  }

  void reserve() {
    vertices.reserve(Nbuffer);
    indices.reserve(Nbuffer);
  }

  void Reserve() {
    Vertices.reserve(Nbuffer);
    indices.reserve(Nbuffer);
  }

// Store the vertex v and its normal vector n.
  GLuint vertex(const triple &v, const triple& n) {
    size_t nvertices=vertices.size();
    vertices.push_back(vertexData(v,n));
    return nvertices;
  }

// Store the vertex v and its normal vector n, without an explicit color.
  GLuint tvertex(const triple &v, const triple& n) {
    size_t nvertices=Vertices.size();
    Vertices.push_back(VertexData(v,n));
    return nvertices;
  }

// Store the vertex v, its normal vector n, and colors c.
  GLuint Vertex(const triple &v, const triple& n, GLfloat *c) {
    size_t nvertices=Vertices.size();
    Vertices.push_back(VertexData(v,n,c));
    return nvertices;
  }

// Store the pixel v and its width.
  GLuint vertex0(const triple &v, double width) {
    size_t nvertices=vertices0.size();
    vertices0.push_back(vertexData0(v,width));
    return nvertices;
  }

  // append array b onto array a with offset
  void appendOffset(std::vector<GLuint>& a,
                    const std::vector<GLuint>& b, size_t offset) {
    size_t n=a.size();
    size_t m=b.size();
    a.resize(n+m);
    for(size_t i=0; i < m; ++i)
      a[n+i]=b[i]+offset;
  }

  void append(const vertexBuffer& b) {
    appendOffset(indices,b.indices,vertices.size());
    vertices.insert(vertices.end(),b.vertices.begin(),b.vertices.end());
  }

  void Append(const vertexBuffer& b) {
    appendOffset(indices,b.indices,Vertices.size());
    Vertices.insert(Vertices.end(),b.Vertices.begin(),b.Vertices.end());
  }

  void append0(const vertexBuffer& b) {
    appendOffset(indices,b.indices,vertices0.size());
    vertices0.insert(vertices0.end(),b.vertices0.begin(),b.vertices0.end());
  }
};

extern vertexBuffer material0Data;   // pixels
extern vertexBuffer material1Data;   // material Bezier curves
extern vertexBuffer materialData;    // material Bezier patches & triangles
extern vertexBuffer colorData;       // colored Bezier patches & triangles
extern vertexBuffer triangleData;    // opaque indexed triangles
extern vertexBuffer transparentData; // transparent patches & triangles

void drawBuffer(vertexBuffer& data, GLint shader, bool color=false);
void drawBuffers();

void clearMaterials();
void clearCenters();

typedef void draw_t();
void setMaterial(vertexBuffer& data, draw_t *draw);

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

  /*
  // Override virtual methods from AsyRender
  void frustum(double left, double right, double bottom,
               double top, double nearVal, double farVal) override;
  void ortho(double left, double right, double bottom,
             double top, double nearVal, double farVal) override;
  */
  void setProjection() override;
  void updateModelViewData() override;
//  void updateProjection() override;

  /** Argument for glrender function - legacy compatibility */
  struct GLRenderArgs: public gc
  {
    string prefix;
    picture* pic;
    string format;
    double width;
    double height;
    double angle;
    double zoom;
    triple m;
    triple M;
    pair shift;
    pair margin;
    double *t;
    double *tup;
    double *background;
    size_t nlights;
    triple *lights;
    double *diffuse;
    double *specular;
    bool view;
  };

  void render(RenderFunctionArgs const& args) override;

  // Legacy glrender function for compatibility - delegates to render()
  static void legacyGlRender(GLRenderArgs const& args, int oldpid=0);

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
  bool initialize = true;
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

  double BBT[9] = {0};

  // Mouse interaction state
  double xprev = 0.0;
  double yprev = 0.0;
  double lastangle = 0.0;
  string currentAction = "";

  // Window state
  bool queueExport = false;
  bool readyAfterExport = false;
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
  void exportHandler(int = 0) override;
  virtual void reshape0(int width, int height) override;
};

// Global OpenGL renderer instance (defined in picture.cc)
extern AsyGLRender* gl;

#endif // HAVE_RENDERER

} // namespace camp

#endif  // GLRENDER_H
