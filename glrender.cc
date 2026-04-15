/*****
 * glrender.cc
 * John Bowman, Orest Shardt, and Supakorn "Jamie" Rassameemasmuang
 * Render 3D Bezier paths and surfaces.
 *****/

#include <stdlib.h>
#include <fstream>
#include <cstring>
#include <cmath>
#include <chrono>
#include <thread>
#include <functional>

#if !defined(_WIN32)
#include <sys/time.h>
#include <unistd.h>
#endif

#include "common.h"
#include "locate.h"
#include "seconds.h"
#include "statistics.h"
#include "bezierpatch.h"
#include "beziercurve.h"

#include "picture.h"
#include "bbox3.h"
#include "drawimage.h"
#include "interact.h"
#include "fpu.h"
#include "renderBase.h"

extern uint32_t CLZ(uint32_t a);

bool GPUindexing = false;  // Disabled by default - compute shaders not needed for opaque rendering
bool GPUcompress;

#ifdef HAVE_RENDERER
// Define extern variables declared in renderBase.h and glrender.h
namespace camp {
glm::dmat4 projViewMat;
glm::dmat3 normMat;  // Normal matrix is 3x3, not 4x4
double glBBT[9] = {0};  // Definition for camp namespace (matches glrender.h)

glm::mat4 viewMat;
const double *dView;
}

#include "tr.h"

#ifdef HAVE_LIBGLFW
#include <GLFW/glfw3.h>
#endif // HAVE_LIBGLFW

#include "shaders.h"
#include "GLTextures.h"
#include "EXRFiles.h"

using settings::locateFile;
using utils::stopWatch;

using namespace settings;
using namespace glm;

#endif // HAVE_RENDERER

#ifdef HAVE_LIBGLM

namespace camp {
Billboard BB;

GLint pixelShader;
GLint materialShader[2];
GLint colorShader[2];
GLint generalShader[2];
GLint countShader;
GLint transparentShader;
GLint blendShader;
GLint zeroShader;
GLint compressShader;
GLint sum1Shader;
GLint sum2Shader;
GLint sum2fastShader;
GLint sum3Shader;

GLuint fragments;

GLuint vao;  // Vertex Array Object
GLuint offsetBuffer;
GLuint indexBuffer;
GLuint elementsBuffer;
GLuint countBuffer;
GLuint globalSumBuffer;
GLuint fragmentBuffer;
GLuint depthBuffer;
GLuint opaqueBuffer;
GLuint opaqueDepthBuffer;
GLuint feedbackBuffer;

bool ssbo;
bool interlock;
}

#endif

#ifdef HAVE_LIBGLM
using camp::Material;
using camp::Maxmaterials;
using camp::Nmaterials;
using camp::nmaterials;
using camp::MaterialMap;

namespace camp {
bool initSSBO;
GLuint maxFragments;

vertexBuffer material0Data(GL_POINTS);
vertexBuffer material1Data(GL_LINES);
vertexBuffer materialData;
vertexBuffer colorData;
vertexBuffer transparentData;
vertexBuffer triangleData;

const size_t Nbuffer=10000;
const size_t nbuffer=1000;

std::vector<Material> materials;
MaterialMap materialMap;
size_t materialIndex;

size_t Maxmaterials;
size_t Nmaterials=1;
size_t nmaterials=48;
unsigned int Opaque=0;

void clearCenters()
{
  camp::drawElement::centers.clear();
  camp::drawElement::centermap.clear();
}

void clearMaterials()
{
  materials.clear();
  materials.reserve(nmaterials);
  materialMap.clear();

  material0Data.partial=false;
  material1Data.partial=false;
  materialData.partial=false;
  colorData.partial=false;
  triangleData.partial=false;
  transparentData.partial=false;
}

} // end namespace camp

// Additional global variables needed by other modules (extern declarations only - definitions are in AsyRender base class)
namespace camp {
extern int fullWidth, fullHeight;
extern bool orthographic_gl;  // Note: different name to avoid conflict with v3dheadertypes::orthographic enum
extern double Angle, Zoom0;
extern camp::pair Shift, Margin;
extern double T[16], Tup[16];
extern double Xmin, Xmax, Ymin, Ymax, Zmin, Zmax;  // These are now member variables in AsyRender
}

extern void exitHandler(int);

using camp::picture;
using camp::drawRawImage;
using camp::transform;
using camp::pair;
using camp::triple;
using vm::array;
using vm::read;
using camp::bbox3;
using settings::getSetting;
using settings::Setting;

// Global variables for cross-translation-unit compatibility (following Vulkan pattern)
// These are kept minimal - most state is in AsyGLRender member variables
namespace camp {
// Minimal globals for shader compatibility (following Vulkan pattern)
bool orthographic_gl = false;
}

// OpenGL-specific global state (minimal set for shader compatibility)
bool Iconify=false;
bool ignorezoom;
int Fitscreen=1;
bool firstFit;
bool queueExport=false;
bool readyAfterExport=false;
bool remesh;
bool copied;
int Mode;
double Aspect;
bool View;
bool ViewExport;
int Oldpid;
string Prefix;
const camp::picture* Picture;
string Format;
int fullWidth,fullHeight;
int Width,Height;
GLuint pixels;
GLuint elements;
GLuint lastpixels;
double oWidth,oHeight;
int screenWidth,screenHeight;
int maxTileWidth;
int maxTileHeight;
double H;
bool haveScene;
double gl_X, gl_Y;
string currentAction="";
double cx,cy;
double xprev,yprev;
static const double ASY_PI=acos(-1.0);
static const double ASY_DEGREES=180.0/ASY_PI;
static const double ASY_RADIANS=1.0/ASY_DEGREES;
size_t Nlights=1;
size_t nlights;
size_t nlights0;
camp::triple *Lights;
double *Diffuse;
double *Specular;
bool antialias;
double zoomFactor = 0.0;
GLint lastshader=-1;
bool format3dWait=false;
unsigned int framecount;

// GPU compute state
GLuint localSize=0;
GLuint blockSize=0;
GLuint groupSize=0;
GLuint g=0;
GLuint maxSize=1;
GLuint processors=0;

template<class T>
inline T min(T a, T b)
{
  return (a < b) ? a : b;
}

template<class T>
inline T max(T a, T b)
{
  return (a > b) ? a : b;
}

glm::vec4 vec4(camp::triple v)
{
  return glm::vec4(v.getx(),v.gety(),v.getz(),0);
}

glm::vec4 vec4(double *v)
{
  return glm::vec4(v[0],v[1],v[2],v[3]);
}

bool Xspin,Yspin,Zspin;

double T[16];
double Tup[16];

#ifdef HAVE_RENDERER

#ifdef HAVE_LIBGLFW
int oldWidth,oldHeight;

bool queueScreen=false;

string Action;

double lastangle;
GLFWwindow* window;
#endif

using utils::statistics;
statistics S;

namespace camp {
// IBL textures - disabled for now due to template issues
void* iblbrdfTex = nullptr;
void* irradianceTex = nullptr;
void* reflTexturesTex = nullptr;
}

gl::GLTexture2<float,GL_FLOAT> fromEXR(string const& EXRFile, gl::GLTexturesFmt const& fmt, GLint const& textureNumber)
{
  camp::IEXRFile fil(EXRFile);
  return gl::GLTexture2<float,GL_FLOAT> {fil.getData(),fil.size(),textureNumber,fmt};
}

gl::GLTexture3<float,GL_FLOAT> fromEXR3(
  mem::vector<string> const& EXRFiles, gl::GLTexturesFmt const& fmt, GLint const& textureNumber)
{
  // 3d reflectance textures
  std::vector<float> data;
  size_t count=EXRFiles.size();
  int wi=0, ht=0;

  for(string const& EXRFile : EXRFiles) {
    camp::IEXRFile fil3(EXRFile);
    std::tie(wi,ht)=fil3.size();
    size_t imSize=4*wi*ht;
    std::copy(fil3.getData(),fil3.getData()+imSize,std::back_inserter(data));
  }

  return gl::GLTexture3<float,GL_FLOAT> {
          data.data(),
          std::tuple<int,int,int>(wi,ht,static_cast<int>(count)),textureNumber,
          fmt
  };
}

void initIBL()
{
  gl::GLTexturesFmt fmt;
  fmt.internalFmt=GL_RGB16F;
  string imageDir=locateFile(getSetting<string>("imageDir"))+"/";
  string imagePath=imageDir+getSetting<string>("image")+"/";
  // IBL textures - disabled for now due to template issues
  // camp::irradianceTex=fromEXR(imagePath+"diffuse.exr",fmt,1);
  // gl::GLTexturesFmt fmtRefl;
  // fmtRefl.internalFmt=GL_RG16F;
  // camp::IBLbrdfTex=fromEXR(imageDir+"refl.exr",fmtRefl,2);

  gl::GLTexturesFmt fmt3;
  fmt3.internalFmt=GL_RGB16F;
  fmt3.wrapS=GL_REPEAT;
  fmt3.wrapR=GL_CLAMP_TO_EDGE;
  fmt3.wrapT=GL_CLAMP_TO_EDGE;

  mem::vector<string> files;
  mem::string prefix=imagePath+"refl";
  for(unsigned int i=0; i <= 10; ++i) {
    mem::stringstream mss;
    mss << prefix << i << ".exr";
    files.emplace_back(mss.str());
  }

  // IBL textures - disabled for now due to template issues
  // camp::reflTexturesTex=fromEXR3(files,fmt3,3);
}

void *glrenderWrapper(void *a);

#ifdef HAVE_LIBOSMESA
OSMesaContext ctx;
unsigned char *osmesa_buffer;
#endif

#ifdef HAVE_PTHREAD

// Note: Pthread synchronization primitives are now members of AsyGLRender class
// to align with Vulkan renderer pattern (vkrender.cc)
// The wait() and endwait() functions are now methods of AsyRender base class

#endif

void noShaders()
{
  cerr << "GLSL shaders not found." << endl;
  exit(-1);
}

void initComputeShaders()
{
  // Ensure context is current before using OpenGL functions
#ifdef HAVE_LIBGLFW
#ifndef HAVE_LIBOSMESA
  if(::window) {
    glfwMakeContextCurrent(::window);
  }
#endif
#endif

  string sum1=locateFile("shaders/sum1.glsl");
  string sum2=locateFile("shaders/sum2.glsl");
  string sum2fast=locateFile("shaders/sum2fast.glsl");
  string sum3=locateFile("shaders/sum3.glsl");

  if(sum1.empty() || sum2.empty() || sum2fast.empty() || sum3.empty())
    noShaders();

  std::vector<ShaderfileModePair> shaders(1);
  std::vector<std::string> shaderParams;

  shaders[0]=ShaderfileModePair(sum1.c_str(),GL_COMPUTE_SHADER);
  ostringstream s,s2;
  s << "LOCALSIZE " << localSize << "u" << endl;
  shaderParams.push_back(s.str().c_str());
  s2 << "BLOCKSIZE " << blockSize << "u" << endl;
  shaderParams.push_back(s2.str().c_str());
  GLuint rc=compileAndLinkShader(shaders,shaderParams,true,false,true,true);
  if(rc == 0) {
    GPUindexing=false; // Compute shaders are unavailable.
    if(settings::verbose > 2)
      cout << "No compute shader support" << endl;
  } else {
//    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT,0,&maxgroups);
//    maxgroups=min(1024,maxgroups/(GLint) (localSize*localSize));
    camp::sum1Shader=rc;

    shaders[0]=ShaderfileModePair(sum2.c_str(),GL_COMPUTE_SHADER);
    camp::sum2Shader=compileAndLinkShader(shaders,shaderParams,true,false,
                                          true);

    shaders[0]=ShaderfileModePair(sum2fast.c_str(),GL_COMPUTE_SHADER);
    camp::sum2fastShader=compileAndLinkShader(shaders,shaderParams,true,false,
                                              true);

    shaders[0]=ShaderfileModePair(sum3.c_str(),GL_COMPUTE_SHADER);
    camp::sum3Shader=compileAndLinkShader(shaders,shaderParams,true,false,
                                          true);
  }
}

void initBlendShader()
{
  string screen=locateFile("shaders/screen.glsl");
  string blend=locateFile("shaders/blend.glsl");

  if(screen.empty() || blend.empty())
    noShaders();

  std::vector<ShaderfileModePair> shaders(2);
  std::vector<std::string> shaderParams;

  ostringstream s;
  s << "ARRAYSIZE " << maxSize << "u" << endl;
  shaderParams.push_back(s.str().c_str());
  if(GPUindexing)
    shaderParams.push_back("GPUINDEXING");
  if(GPUcompress)
    shaderParams.push_back("GPUCOMPRESS");
  shaders[0]=ShaderfileModePair(screen.c_str(),GL_VERTEX_SHADER);
  shaders[1]=ShaderfileModePair(blend.c_str(),GL_FRAGMENT_SHADER);
  camp::blendShader=compileAndLinkShader(shaders,shaderParams,camp::ssbo);
}

// Return the smallest power of 2 greater than or equal to n.
inline GLuint ceilpow2(GLuint n)
{
  --n;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  return ++n;
}

void initShaders()
{
  // Ensure context is current before using OpenGL functions
#ifdef HAVE_LIBGLFW
#ifndef HAVE_LIBOSMESA
  if(::window) {
    glfwMakeContextCurrent(::window);
  }
#endif
#endif

  Nlights=nlights == 0 ? 0 : std::max(Nlights,nlights);
  Nmaterials=std::max(Nmaterials,nmaterials);

  string zero=locateFile("shaders/zero.glsl");
  string compress=locateFile("shaders/compress.glsl");
  string vertex=locateFile("shaders/vertex.glsl");
  string count=locateFile("shaders/count.glsl");
  string fragment=locateFile("shaders/fragment.glsl");
  string screen=locateFile("shaders/screen.glsl");

  if(zero.empty() || compress.empty() || vertex.empty() || fragment.empty() ||
     screen.empty() || count.empty())
    noShaders();

  // Only try compute shaders if GPUindexing is explicitly enabled and we have a valid context
  // Note: GPUindexing defaults to false as compute shaders are not needed for opaque rendering
  if(GPUindexing) {
#ifdef HAVE_LIBGLFW
#ifndef HAVE_LIBOSMESA
    // Check if we have a valid OpenGL context before trying compute shaders
    if(::window && glfwGetCurrentContext() == ::window) {
      initComputeShaders();
    } else {
      GPUindexing = false; // Disable if no valid context
      if(settings::verbose > 2)
        cout << "No valid OpenGL context for compute shaders" << endl;
    }
#else
    initComputeShaders();
#endif
#else
    initComputeShaders();
#endif
  }

  std::vector<ShaderfileModePair> shaders(2);
  std::vector<std::string> shaderParams;

  if(camp::glRenderer && camp::glRenderer->ibl) {
    shaderParams.push_back("USE_IBL");
    initIBL();
  }

  shaders[0]=ShaderfileModePair(vertex.c_str(),GL_VERTEX_SHADER);

#ifdef HAVE_SSBO
  if(GPUindexing)
    shaderParams.push_back("GPUINDEXING");
  if(GPUcompress)
    shaderParams.push_back("GPUCOMPRESS");
  shaders[1]=ShaderfileModePair(count.c_str(),GL_FRAGMENT_SHADER);
  camp::countShader=compileAndLinkShader(shaders,shaderParams,
                                         true,false,false,true);
  if(camp::countShader)
    shaderParams.push_back("HAVE_SSBO");
#else
  camp::countShader=0;
#endif

  camp::ssbo=camp::countShader;
#ifdef HAVE_LIBOSMESA
  camp::interlock=false;
#else
  camp::interlock=camp::ssbo && getSetting<bool>("GPUinterlock");
#endif

  if(!camp::ssbo && settings::verbose > 2)
    cout << "No SSBO support; order-independent transparency unavailable"
         << endl;

  shaders[1]=ShaderfileModePair(fragment.c_str(),GL_FRAGMENT_SHADER);
  shaderParams.push_back("MATERIAL");
  if(camp::orthographic_gl)
    shaderParams.push_back("ORTHOGRAPHIC");

  ostringstream lights,materials,opaque;
  lights << "Nlights " << Nlights;
  shaderParams.push_back(lights.str().c_str());
  materials << "Nmaterials " << Nmaterials;
  shaderParams.push_back(materials.str().c_str());

  shaderParams.push_back("WIDTH");
  camp::pixelShader=compileAndLinkShader(shaders,shaderParams,camp::ssbo);
  shaderParams.pop_back();

  shaderParams.push_back("NORMAL");
  if(camp::interlock) shaderParams.push_back("HAVE_INTERLOCK");
  camp::materialShader[0]=compileAndLinkShader(shaders,shaderParams,
                                               camp::ssbo,camp::interlock,false,true);
  if(camp::interlock && !camp::materialShader[0]) {
    shaderParams.pop_back();
    camp::interlock=false;
    camp::materialShader[0]=compileAndLinkShader(shaders,shaderParams,camp::ssbo);
    if(settings::verbose > 2)
      cout << "No fragment shader interlock support" << endl;
  }

  shaderParams.push_back("OPAQUE");
  camp::materialShader[1]=compileAndLinkShader(shaders,shaderParams,camp::ssbo);
  shaderParams.pop_back();

  shaderParams.push_back("COLOR");
  camp::colorShader[0]=compileAndLinkShader(shaders,shaderParams,camp::ssbo,
                                            camp::interlock);
  shaderParams.push_back("OPAQUE");
  camp::colorShader[1]=compileAndLinkShader(shaders,shaderParams,camp::ssbo);
  shaderParams.pop_back();

  shaderParams.push_back("GENERAL");
  if(Mode != 0)
    shaderParams.push_back("WIREFRAME");
  camp::generalShader[0]=compileAndLinkShader(shaders,shaderParams,camp::ssbo,
                                              camp::interlock);
  shaderParams.push_back("OPAQUE");
  camp::generalShader[1]=compileAndLinkShader(shaders,shaderParams,camp::ssbo);
  shaderParams.pop_back();

  shaderParams.push_back("TRANSPARENT");
  camp::transparentShader=compileAndLinkShader(shaders,shaderParams,camp::ssbo,
                                               camp::interlock);
  shaderParams.clear();

  if(camp::ssbo) {
    if(GPUindexing)
      shaderParams.push_back("GPUINDEXING");
    shaders[0]=ShaderfileModePair(screen.c_str(),GL_VERTEX_SHADER);
    shaders[1]=ShaderfileModePair(compress.c_str(),GL_FRAGMENT_SHADER);
    camp::compressShader=compileAndLinkShader(shaders,shaderParams,camp::ssbo);
    if(GPUindexing)
      shaderParams.pop_back();
    else {
      shaders[1]=ShaderfileModePair(zero.c_str(),GL_FRAGMENT_SHADER);
      camp::zeroShader=compileAndLinkShader(shaders,shaderParams,camp::ssbo);
    }
    maxSize=1;
    initBlendShader();
  }
  lastshader=-1;
}

void deleteComputeShaders()
{
  glDeleteProgram(camp::sum1Shader);
  glDeleteProgram(camp::sum2Shader);
  glDeleteProgram(camp::sum2fastShader);
  glDeleteProgram(camp::sum3Shader);
}

void deleteBlendShader()
{
  glDeleteProgram(camp::blendShader);
}

void deleteShaders()
{
  if(camp::ssbo) {
    deleteBlendShader();
    if(GPUindexing)
      deleteComputeShaders();
    else
      glDeleteProgram(camp::zeroShader);
    glDeleteProgram(camp::countShader);
    glDeleteProgram(camp::compressShader);
  }

  if (camp::transparentShader != 0)
    glDeleteProgram(camp::transparentShader);
  for(unsigned int opaque=0; opaque < 2; ++opaque) {
    if (camp::generalShader[opaque] != 0)
      glDeleteProgram(camp::generalShader[opaque]);
    if (camp::colorShader[opaque] != 0)
      glDeleteProgram(camp::colorShader[opaque]);
    if (camp::materialShader[opaque] != 0)
      glDeleteProgram(camp::materialShader[opaque]);
  }
  if (camp::pixelShader != 0)
    glDeleteProgram(camp::pixelShader);
}

void resizeBlendShader(GLuint maxsize)
{
  maxSize=ceilpow2(maxsize);
  deleteBlendShader();
  initBlendShader();
}

void setBuffers()
{
  if(settings::verbose > 2) {
    cerr << "setBuffers: Creating VAO, camp::vao=" << camp::vao << endl;
  }
  glGenVertexArrays(1,&camp::vao);
  if(settings::verbose > 2) {
    cerr << "setBuffers: VAO created, camp::vao=" << camp::vao << endl;
  }
  // Bind VAO once and leave it bound for all subsequent draw operations
  glBindVertexArray(camp::vao);

  camp::material0Data.reserve0();
  camp::materialData.reserve();
  camp::colorData.Reserve();
  camp::triangleData.Reserve();
  camp::transparentData.Reserve();

#ifdef HAVE_SSBO
  glGenBuffers(1, &camp::offsetBuffer);
  if(GPUindexing)
    glGenBuffers(1, &camp::globalSumBuffer);
  glGenBuffers(1, &camp::feedbackBuffer);
  glGenBuffers(1, &camp::countBuffer);
  if(GPUcompress) {
    glGenBuffers(1, &camp::indexBuffer);
    glGenBuffers(1, &camp::elementsBuffer);
  }
  glGenBuffers(1, &camp::fragmentBuffer);
  glGenBuffers(1, &camp::depthBuffer);
  glGenBuffers(1, &camp::opaqueBuffer);
  glGenBuffers(1, &camp::opaqueDepthBuffer);
#endif

  if(settings::verbose > 2) {
    cerr << "setBuffers: Done, camp::vao=" << camp::vao << endl;
  }
}

bool exporting=false;

void drawscene(int Width, int Height)
{
  // Access all state through the renderer instance (following Vulkan pattern)
  camp::AsyGLRender* glr = camp::glRenderer;
  if(!glr) return;

#ifdef HAVE_PTHREAD
  static bool first=true;
  if(glr->renderThread && first) {
    glr->wait(glr->initSignal,glr->initLock);
    glr->endwait(glr->initSignal,glr->initLock);
    first=false;
  }

  if(format3dWait)
    glr->wait(glr->initSignal,glr->initLock);
#endif

#ifdef HAVE_LIBGLFW
#ifndef HAVE_LIBOSMESA
  // Diagnostics for debugging segfault
  if(settings::verbose > 2) {
    cerr << "drawscene: Width=" << Width << " Height=" << Height
         << " window=" << (glr ? glr->glfwWindow : nullptr)
         << " current context=" << glfwGetCurrentContext() << endl;
  }
#endif
#endif

  if((nlights == 0 && Nlights > 0) || nlights > Nlights ||
     nmaterials > Nmaterials) {
    deleteShaders();
    initShaders();
  }

  if(settings::verbose > 2) {
    cerr << "drawscene: about to call glClear..." << endl;
  }

  // Set viewport before clearing (in case it wasn't set)
  // Skip during export - trBeginTile handles viewport for tiling
  if(!exporting)
    glViewport(0, 0, Width, Height);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // Use member variables from AsyGLRender (following Vulkan pattern)
  cerr << "DEBUG drawscene: xmin=" << glr->xmin << " xmax=" << glr->xmax
       << " ymin=" << glr->ymin << " ymax=" << glr->ymax
       << " Zmin=" << glr->Zmin << " Zmax=" << glr->Zmax << endl;
  if(glr->xmin >= glr->xmax || glr->ymin >= glr->ymax || glr->Zmin >= glr->Zmax) return;

  triple m(glr->xmin,glr->ymin,glr->Zmin);
  triple M(glr->xmax,glr->ymax,glr->Zmax);
  double perspective=glr->orthographic || glr->Zmax == 0.0 ? 0.0 : 1.0/glr->Zmax;

  double size2=hypot(Width,Height);

  cerr << "DEBUG: m=(" << m.getx() << "," << m.gety() << "," << m.getz() << ") M=(" << M.getx() << "," << M.gety() << "," << M.getz() << ")" << endl;
  cerr << "DEBUG: Picture=" << (void*)Picture << " size2=" << size2 << " perspective=" << perspective << endl;

  if(remesh)
    camp::clearCenters();

  if(settings::verbose > 2) {
    cerr << "drawscene: calling Picture->render()" << endl;
  }
  if(Picture)
    Picture->render(size2,m,M,perspective,remesh);

  if(settings::verbose > 2) {
    cerr << "drawscene: Picture->render() complete" << endl;
  }

#ifdef HAVE_RENDERER
  camp::drawBuffers();
#endif

  if(!camp::glRenderer || !camp::glRenderer->outlinemode) remesh=false;
}

// Return x divided by y rounded up to the nearest integer.
int ceilquotient(int x, int y)
{
  return (x+y-1)/y;
}

void Export()
{
  size_t ndata=3*fullWidth*fullHeight;
  if(ndata == 0) return;
  glReadBuffer(GL_BACK_LEFT);
  glPixelStorei(GL_PACK_ALIGNMENT,1);
  glFinish();
  exporting=true;

  try {
    unsigned char *data=new unsigned char[ndata];
    if(data) {
      TRcontext *tr=trNew();
      int width=ceilquotient(fullWidth,
                             ceilquotient(fullWidth,std::min(maxTileWidth,Width)));
      int height=ceilquotient(fullHeight,
                              ceilquotient(fullHeight,
                                           std::min(maxTileHeight,Height)));
      if(settings::verbose > 1)
        cout << "Exporting " << Prefix << " as " << fullWidth << "x"
             << fullHeight << " image" << " using tiles of size "
             << width << "x" << height << endl;

      unsigned border=std::min(std::min(1,(width-1)/2),(height-1)/2);
      trTileSize(tr,width,height,border);
      trImageSize(tr,fullWidth,fullHeight);
      trImageBuffer(tr,GL_RGB,GL_UNSIGNED_BYTE,data);

      // Note: setDimensions is protected, use global state directly
      // Width and Height are already set above via reshape0()

      // Use member variables from AsyGLRender (following Vulkan pattern)
      double dXmin = camp::glRenderer->xmin;
      double dXmax = camp::glRenderer->xmax;
      double dYmin = camp::glRenderer->ymin;
      double dYmax = camp::glRenderer->ymax;
      double dZmin = camp::glRenderer->Zmin;
      double dZmax = camp::glRenderer->Zmax;

      size_t count=0;
      if(haveScene) {
        (camp::orthographic_gl ? trOrtho : trFrustum)(tr,dXmin,dXmax,dYmin,dYmax,-dZmax,-dZmin);
        do {
          trBeginTile(tr);
          remesh=true;
          drawscene(fullWidth,fullHeight);
          lastshader=-1;
          ++count;
        } while (trEndTile(tr));
      } else {// clear screen and return
        drawscene(fullWidth,fullHeight);
      }

      if(settings::verbose > 1)
        cout << count << " tile" << (count != 1 ? "s" : "") << " drawn" << endl;
      trDelete(tr);

      picture pic;
      drawRawImage *Image=NULL;
      if(haveScene) {
        double w=oWidth;
        double h=oHeight;
        double Aspect=((double) fullWidth)/fullHeight;
        if(w > h*Aspect) w=(int) (h*Aspect+0.5);
        else h=(int) (w/Aspect+0.5);
        // Render an antialiased image.

        Image=new drawRawImage(data,fullWidth,fullHeight,
                               transform(0.0,0.0,w,0.0,0.0,h),
                               antialias);
        pic.append(Image);
      }

      pic.shipout(NULL,Prefix,Format,false,ViewExport);
      if(Image)
        delete Image;
      delete[] data;
    }
  } catch(handled_error const&) {
  } catch(std::bad_alloc&) {
    outOfMemory();
  }
  remesh=true;
  // Note: setProjection is protected, will be called from class methods

#ifndef HAVE_LIBOSMESA
#ifdef HAVE_LIBGLFW
  if(camp::glRenderer) camp::glRenderer->redraw=true;
#endif

#ifdef HAVE_PTHREAD
  if(camp::glRenderer && camp::glRenderer->renderThread && readyAfterExport) {
    readyAfterExport=false;
    camp::glRenderer->endwait(camp::glRenderer->readySignal,camp::glRenderer->readyLock);
  }
#endif
#endif
  exporting=false;
  camp::initSSBO=true;
}

void nodisplay()
{
}

// destroywindow is no longer needed with GLFW (window destruction is handled directly)

// Return the greatest power of 2 less than or equal to n.
inline unsigned int floorpow2(unsigned int n)
{
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  return n-(n >> 1);
}

void quit()
{
#ifdef HAVE_LIBOSMESA
  if(osmesa_buffer) delete[] osmesa_buffer;
  if(ctx) OSMesaDestroyContext(ctx);
  exit(0);
#endif
#ifdef HAVE_LIBGLFW
  if(camp::glRenderer && camp::glRenderer->renderThread) {
    camp::glRenderer->home();
#ifdef HAVE_PTHREAD
    if(!interact::interactive) {
      // idle() is no longer needed with GLFW event loop
      camp::glRenderer->endwait(camp::glRenderer->readySignal,camp::glRenderer->readyLock);
    }
#endif
    // Always signal the window to close in threaded mode
    if (camp::glRenderer->glfwWindow != nullptr) {
      glfwSetWindowShouldClose(static_cast<GLFWwindow*>(camp::glRenderer->glfwWindow), true);
      if(interact::interactive) {
        glfwHideWindow(static_cast<GLFWwindow*>(camp::glRenderer->glfwWindow));
      }
    }
  } else {
    if(camp::glRenderer && camp::glRenderer->glfwWindow != nullptr) {
      glfwDestroyWindow(static_cast<GLFWwindow*>(camp::glRenderer->glfwWindow));
    }
    glfwTerminate();
    exit(0);
  }
#else
  // No windowing system available - just exit
  exit(0);
#endif
}

void mode()
{
  remesh=true;
  if(camp::ssbo)
    camp::initSSBO=true;
  ++Mode;
  if(Mode > 2) Mode=0;

  switch(Mode) {
    case 0: // regular
      if(camp::glRenderer) camp::glRenderer->outlinemode=false;
      if(camp::glRenderer) camp::glRenderer->ibl=getSetting<bool>("ibl");
      nlights=nlights0;
      lastshader=-1;
      glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
      break;
    case 1: // outline
      if(camp::glRenderer) camp::glRenderer->outlinemode=true;
      if(camp::glRenderer) camp::glRenderer->ibl=false;
      nlights=0; // Force shader recompilation
      glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
      break;
    case 2: // wireframe
      if(camp::glRenderer) camp::glRenderer->outlinemode=false;
      Nlights=1; // Force shader recompilation
      break;
  }
#ifndef HAVE_LIBOSMESA
  if(camp::glRenderer) camp::glRenderer->redraw=true;
#endif
}

// GUI-related functions
#ifdef HAVE_LIBGLFW
bool capsize(int& width, int& height)
{
  bool resize=false;
  if(width > screenWidth) {
    width=screenWidth;
    resize=true;
  }
  if(height > screenHeight) {
    height=screenHeight;
    resize=true;
  }
  return resize;
}

void reshape0(int width, int height)
{
  gl_X=(gl_X/Width)*width;
  gl_Y=(gl_Y/Height)*height;

  Width=width;
  Height=height;

  static int lastWidth=1;
  static int lastHeight=1;
  if(View && Width*Height > 1 && (Width != lastWidth || Height != lastHeight)
     && settings::verbose > 1) {
    cout << "Rendering " << stripDir(Prefix) << " as "
         << Width << "x" << Height << " image" << endl;
    lastWidth=Width;
    lastHeight=Height;
  }

  // Note: setProjection is protected, will be called from class methods
  glViewport(0,0,Width,Height);
  if(camp::ssbo)
    camp::initSSBO=true;
}

void windowposition(int& x, int& y, int width=Width, int height=Height)
{
  pair z=getSetting<pair>("position");
  x=(int) z.getx();
  y=(int) z.gety();
  if(x < 0) {
    x += screenWidth-width;
    if(x < 0) x=0;
  }
  if(y < 0) {
    y += screenHeight-height;
    if(y < 0) y=0;
  }
}

void setsize(int w, int h, bool reposition=true)
{
  int x,y;

  capsize(w,h);

  // Use window from glRenderer if available, otherwise use global window variable
  GLFWwindow* win = nullptr;
  if (camp::glRenderer && camp::glRenderer->glfwWindow != nullptr) {
    win = static_cast<GLFWwindow*>(camp::glRenderer->glfwWindow);
  } else {
    win = ::window;
  }

  if (win == nullptr) return;

  if(reposition) {
    windowposition(x,y,w,h);
    glfwSetWindowPos(win,x,y);
  } else {
    int wx, wy;
    glfwGetWindowPos(win, &wx, &wy);
    glfwSetWindowPos(win,std::max(wx-2,0),std::max(wy-2,0));
  }

  glfwSetWindowSize(win,w,h);
  reshape0(w,h);
  if(camp::glRenderer) camp::glRenderer->redraw=true;
}

void capzoom()
{
  static double maxzoom=sqrt(DBL_MAX);
  static double minzoom=1.0/maxzoom;
  camp::AsyGLRender* glr = camp::glRenderer;
  if(!glr) return;
  if(glr->Zoom <= minzoom) glr->Zoom=minzoom;
  if(glr->Zoom >= maxzoom) glr->Zoom=maxzoom;

  if(fabs(glr->Zoom-glr->lastzoom) > settings::getSetting<double>("zoomThreshold")) {
    remesh=true;
    glr->lastzoom=glr->Zoom;
  }
}

void fullscreen(bool reposition=true)
{
  camp::AsyGLRender* glr = camp::glRenderer;
  if(!glr) return;
  Width=screenWidth;
  Height=screenHeight;
  if(firstFit) {
    if(Width < Height*Aspect)
      glr->Zoom *= Width/(Height*Aspect);
    capzoom();
    // Note: setProjection is protected, will be called from class methods
    firstFit=false;
  }
  glr->Xfactor=((double) screenHeight)/Height;
  glr->Yfactor=((double) screenWidth)/Width;
  reshape0(Width,Height);

  // Use window from glRenderer if available, otherwise use global window variable
  GLFWwindow* win = nullptr;
  if (camp::glRenderer && camp::glRenderer->glfwWindow != nullptr) {
    win = static_cast<GLFWwindow*>(camp::glRenderer->glfwWindow);
  } else {
    win = ::window;
  }

  if (win != nullptr) {
    if(reposition)
      glfwSetWindowPos(win,0,0);
    glfwSetWindowSize(win,Width,Height);
  }

  if(camp::glRenderer) camp::glRenderer->redraw=true;
}

void fitscreen(bool reposition=true)
{
  camp::AsyGLRender* glr = camp::glRenderer;
  if(!glr) return;
  switch(Fitscreen) {
    case 0: // Original size
    {
      glr->Xfactor=glr->Yfactor=1.0;
      double pixelRatio=getSetting<double>("devicepixelratio");
      setsize(oldWidth*pixelRatio,oldHeight*pixelRatio,reposition);
      break;
    }
    case 1: // Fit to screen in one dimension
    {
      int w=screenWidth;
      int h=screenHeight;
      if(w > h*Aspect)
        w=std::min((int) ceil(h*Aspect),w);
      else
        h=std::min((int) ceil(w/Aspect),h);
      setsize(w,h,reposition);
      break;
    }
    case 2: // Full screen
    {
      fullscreen(reposition);
      break;
    }
  }
}

void togglefitscreen()
{
  ++Fitscreen;
  if(Fitscreen > 2) Fitscreen=0;
  fitscreen();
}

void screen()
{
  if(camp::glRenderer && camp::glRenderer->renderThread && !interact::interactive)
    fitscreen(false);
}

stopWatch frameTimer;

void nextframe()
{
#ifdef HAVE_PTHREAD
  if(camp::glRenderer) {
    camp::glRenderer->endwait(camp::glRenderer->readySignal,camp::glRenderer->readyLock);
  }
#endif
  double delay=getSetting<double>("framerate");
  if(delay != 0.0) delay=1.0/delay;
  double seconds=frameTimer.seconds(true);
  delay -= seconds;
  if(delay > 0) {
    std::this_thread::sleep_for(std::chrono::duration<double>(delay));
  }
}

stopWatch Timer;

void display()
{
  if(queueScreen) {
    screen();
    queueScreen=false;
  }

  bool fps=settings::verbose > 2;
  drawscene(Width,Height);
  if(fps) {
    if(framecount < 20) // Measure steady-state framerate
      Timer.reset();
    else {
      double s=Timer.seconds(true);
      if(s > 0.0) {
        double rate=1.0/s;
        S.add(rate);
        if(framecount % 20 == 0)
          cout << "FPS=" << rate << "\t" << S.mean() << " +/- " << S.stdev()
               << endl;
      }
    }
    ++framecount;
  }

  // Use window from glRenderer if available, otherwise use global window variable
  GLFWwindow* win = nullptr;
  if (camp::glRenderer && camp::glRenderer->glfwWindow != nullptr) {
    win = static_cast<GLFWwindow*>(camp::glRenderer->glfwWindow);
  } else {
    win = ::window;
  }

  if (win != nullptr) {
    glfwSwapBuffers(win);
  }

  if(queueExport) {
    Export();
    queueExport=false;
  }
  if(!camp::glRenderer || !camp::glRenderer->renderThread) {
#if !defined(_WIN32)
    if(Oldpid != 0 && waitpid(Oldpid,NULL,WNOHANG) != Oldpid) {
      kill(Oldpid,SIGHUP);
      Oldpid=0;
    }
#endif
  }
}

void camp::AsyGLRender::update()
{
  capzoom();

  if(camp::glRenderer) camp::glRenderer->redraw=true;
#ifdef HAVE_RENDERER
  if(camp::glRenderer->glfwWindow) ::glfwShowWindow(static_cast<GLFWwindow*>(camp::glRenderer->glfwWindow));
#endif
  // Use member variables from the renderer instance (matching reference GLUT code)
  double dZmin = camp::glRenderer->Zmin;
  double dZmax = camp::glRenderer->Zmax;
  double cz=0.5*(dZmin+dZmax);

  // Match reference: two translations - first to center, then back along Z
  // Use member variables cx, cy, rotateMat which are updated during interaction
  camp::viewMat = translate(translate(dmat4(1.0), dvec3(camp::glRenderer->cx, camp::glRenderer->cy, cz)) * camp::glRenderer->rotateMat,
                            dvec3(0, 0, -cz));

  // Sync member viewMat and update projection matrices
  camp::glRenderer->viewMat = glm::dmat4(camp::viewMat);
  setProjection();
  camp::glRenderer->updateModelViewData();
}

void updateHandler(int)
{
  queueScreen=true;
  remesh=true;
  camp::glRenderer->update();

  // Use window from glRenderer if available, otherwise use global window variable
  GLFWwindow* win = nullptr;
  if (camp::glRenderer && camp::glRenderer->glfwWindow != nullptr) {
    win = static_cast<GLFWwindow*>(camp::glRenderer->glfwWindow);
  } else {
    win = ::window;
  }

  if (win != nullptr) {
    glfwShowWindow(win);
  }
}

// poll is no longer needed with GLFW - event handling is done in the main loop

void reshape(int width, int height)
{
  if(camp::glRenderer && camp::glRenderer->renderThread) {
    static bool initialize=true;
    if(initialize) {
      initialize=false;
#if !defined(_WIN32)
      Signal(SIGUSR1,updateHandler);
#endif
    }
  }

  // Use window from glRenderer if available, otherwise use global window variable
  GLFWwindow* win = nullptr;
  if (camp::glRenderer && camp::glRenderer->glfwWindow != nullptr) {
    win = static_cast<GLFWwindow*>(camp::glRenderer->glfwWindow);
  } else {
    win = ::window;
  }

  if(capsize(width,height)) {
    if (win != nullptr) {
      glfwSetWindowSize(win,width,height);
    }
  }

  reshape0(width,height);
  remesh=true;
}

void exportHandler(int=0)
{
#ifndef HAVE_LIBOSMESA
#ifdef HAVE_LIBGLFW
  if(!Iconify) {
    // Use window from glRenderer if available, otherwise use global window variable
    GLFWwindow* win = nullptr;
    if (camp::glRenderer && camp::glRenderer->glfwWindow != nullptr) {
      win = static_cast<GLFWwindow*>(camp::glRenderer->glfwWindow);
    } else {
      win = ::window;
    }

    if (win != nullptr) {
      glfwShowWindow(win);
    }
  }
#endif
#endif
  readyAfterExport=true;
  Export();

#ifndef HAVE_LIBOSMESA
#ifdef HAVE_LIBGLFW
  if(!Iconify)
    glfwHideWindow(window);
#endif
#endif
}

void init_osmesa()
{
#ifdef HAVE_LIBOSMESA
  // create context and buffer
  if(settings::verbose > 1)
    cout << "Allocating osmesa_buffer of size " << screenWidth << "x"
         << screenHeight << "x4x" << sizeof(GLubyte) << endl;
  osmesa_buffer=new unsigned char[screenWidth*screenHeight*4*sizeof(GLubyte)];
  if(!osmesa_buffer) {
    cerr << "Cannot allocate image buffer." << endl;
    exit(-1);
  }

  const int attribs[]={
    OSMESA_FORMAT,OSMESA_RGBA,
    OSMESA_DEPTH_BITS,16,
    OSMESA_STENCIL_BITS,0,
    OSMESA_ACCUM_BITS,0,
    OSMESA_PROFILE,OSMESA_COMPAT_PROFILE,
    OSMESA_CONTEXT_MAJOR_VERSION,4,
    OSMESA_CONTEXT_MINOR_VERSION,3,
    0,0
  };

  ctx=OSMesaCreateContextAttribs(attribs,NULL);
  if(!ctx) {
    ctx=OSMesaCreateContextExt(OSMESA_RGBA,16,0,0,NULL);
    if(!ctx) {
      cerr << "OSMesaCreateContextExt failed." << endl;
      exit(-1);
    }
  }

  if(!OSMesaMakeCurrent(ctx,osmesa_buffer,GL_UNSIGNED_BYTE,
                        screenWidth,screenHeight )) {
    cerr << "OSMesaMakeCurrent failed." << endl;
    exit(-1);
  }

  int z=0, s=0, a=0;
  glGetIntegerv(GL_DEPTH_BITS,&z);
  glGetIntegerv(GL_STENCIL_BITS,&s);
  glGetIntegerv(GL_ACCUM_RED_BITS,&a);
  if(settings::verbose > 1)
    cout << "Offscreen context settings: Depth=" << z << " Stencil=" << s
         << " Accum=" << a << endl;

  if(z <= 0) {
    cerr << "Error initializing offscreen context: Depth=" << z << endl;
    exit(-1);
  }
#endif // HAVE_LIBOSMESA
}

bool NVIDIA()
{
#ifdef GL_SHADING_LANGUAGE_VERSION
  const char *GLSL_VERSION=(const char *)
    glGetString(GL_SHADING_LANGUAGE_VERSION);
#else
  const char *GLSL_VERSION="";
#endif
  return string(GLSL_VERSION).find("NVIDIA") != string::npos;
}

#endif /* HAVE_RENDERER */

namespace camp {

string getLightIndex(size_t const& index, string const& fieldName) {
  ostringstream buf;
  buf << "lights[" << index << "]." << fieldName;
  return Strdup(buf.str());
}

string getCenterIndex(size_t const& index) {
  ostringstream buf;
  buf << "Centers[" << index << "]";
  return Strdup(buf.str());
}

template<class T>
void registerBuffer(const std::vector<T>& buffervector, GLuint& bufferIndex,
                    bool copy, GLenum type=GL_ARRAY_BUFFER) {
  if(!buffervector.empty()) {
    if(bufferIndex == 0) {
      glGenBuffers(1,&bufferIndex);
      copy=true;
    }
    glBindBuffer(type,bufferIndex);
    if(copy)
      glBufferData(type,buffervector.size()*sizeof(T),
                   buffervector.data(),GL_STATIC_DRAW);
  }
}

void clearCount()
{
  glUseProgram(zeroShader);
  lastshader=zeroShader;
  glUniform1ui(glGetUniformLocation(zeroShader,"width"),Width);
  fpu_trap(false); // Work around FE_INVALID
  glDrawArrays(GL_TRIANGLES, 0, 3);
  fpu_trap(settings::trap());
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void compressCount()
{
  glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
  glUseProgram(compressShader);
  lastshader=compressShader;
  glUniform1ui(glGetUniformLocation(compressShader,"width"),Width);
  fpu_trap(false); // Work around FE_INVALID
  glDrawArrays(GL_TRIANGLES, 0, 3);
  fpu_trap(settings::trap());
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void partialSums(bool readSize=false)
{
  // Compute partial sums on the GPU
  glUseProgram(sum1Shader);
  glDispatchCompute(g,1,1);

  if(elements <= groupSize*groupSize)
    glUseProgram(sum2fastShader);
  else {
    glUseProgram(sum2Shader);
    glUniform1ui(glGetUniformLocation(sum2Shader,"blockSize"),
                 ceilquotient(g,localSize));
  }
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
  glDispatchCompute(1,1,1);

  glUseProgram(sum3Shader);
  glUniform1ui(glGetUniformLocation(sum3Shader,"final"),elements-1);
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
  glDispatchCompute(g,1,1);
}

void resizeFragmentBuffer()
{
  if(GPUindexing) {
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER,camp::feedbackBuffer);
    GLuint *feedback=(GLuint *) glMapBuffer(GL_SHADER_STORAGE_BUFFER,GL_READ_ONLY);

    GLuint maxDepth=feedback[0];
    if(maxDepth > maxSize)
      resizeBlendShader(maxDepth);

    fragments=feedback[1];
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
  }

  if(fragments > maxFragments) {
    // Initialize the alpha buffer
    maxFragments=11*fragments/10;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER,camp::fragmentBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER,maxFragments*sizeof(glm::vec4),
                 NULL,GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,4,camp::fragmentBuffer);


    glBindBuffer(GL_SHADER_STORAGE_BUFFER,camp::depthBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER,maxFragments*sizeof(GLfloat),
                 NULL,GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,5,camp::depthBuffer);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER,camp::feedbackBuffer);
  }
}

void refreshBuffers()
{
  GLuint zero=0;
  pixels=(Width+1)*(Height+1);

  if(initSSBO) {
    processors=1;

    GLuint Pixels;
    if(GPUindexing) {
      GLuint G=ceilquotient(pixels,groupSize);
      Pixels=groupSize*G;

      GLuint globalSize=localSize*ceilquotient(G,localSize);
      glBindBuffer(GL_SHADER_STORAGE_BUFFER,camp::globalSumBuffer);
      glBufferData(GL_SHADER_STORAGE_BUFFER,globalSize*sizeof(GLuint),NULL,
                   GL_DYNAMIC_READ);
      glClearBufferData(GL_SHADER_STORAGE_BUFFER,GL_R32UI,GL_RED_INTEGER,
                        GL_UNSIGNED_INT,&zero);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER,3,camp::globalSumBuffer);
    } else Pixels=pixels;

    glBindBuffer(GL_SHADER_STORAGE_BUFFER,camp::offsetBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER,(Pixels+2)*sizeof(GLuint),
                 NULL,GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,camp::offsetBuffer);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER,camp::countBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER,(Pixels+2)*sizeof(GLuint),
                 NULL,GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,2,camp::countBuffer);

    if(GPUcompress) {
      GLuint one=1;
      glBindBuffer(GL_ATOMIC_COUNTER_BUFFER,camp::elementsBuffer);
      glBufferData(GL_ATOMIC_COUNTER_BUFFER,sizeof(GLuint),&one,
                   GL_DYNAMIC_DRAW);
      glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER,0,camp::elementsBuffer);

      glBindBuffer(GL_SHADER_STORAGE_BUFFER,camp::indexBuffer);
      glBufferData(GL_SHADER_STORAGE_BUFFER,pixels*sizeof(GLuint),
                   NULL,GL_DYNAMIC_DRAW);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER,1,camp::indexBuffer);
    }
    glClearBufferData(GL_SHADER_STORAGE_BUFFER,GL_R32UI,GL_RED_INTEGER,
                      GL_UNSIGNED_INT,&zero); // Clear count or index buffer

    glBindBuffer(GL_SHADER_STORAGE_BUFFER,camp::opaqueBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER,pixels*sizeof(glm::vec4),NULL,
                 GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,6,camp::opaqueBuffer);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER,camp::opaqueDepthBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 sizeof(GLuint)+pixels*sizeof(GLfloat),NULL,
                 GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,7,camp::opaqueDepthBuffer);
    const GLfloat zerof=0.0;
    glClearBufferData(GL_SHADER_STORAGE_BUFFER,GL_R32F,GL_RED,GL_FLOAT,&zerof);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER,camp::feedbackBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER,2*sizeof(GLuint),NULL,
                 GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,8,camp::feedbackBuffer);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER,camp::feedbackBuffer);
    initSSBO=false;
  }

  // Determine the fragment offsets

  if(exporting && GPUindexing && !GPUcompress) {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER,camp::countBuffer);
    glClearBufferData(GL_SHADER_STORAGE_BUFFER,GL_R32UI,GL_RED_INTEGER,
                      GL_UNSIGNED_INT,&zero);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER,camp::feedbackBuffer);
  }

  if(!interlock) {
    drawBuffer(material1Data,countShader);
    drawBuffer(materialData,countShader);
    drawBuffer(colorData,countShader,true);
    drawBuffer(triangleData,countShader,true);
  }

  glDepthMask(GL_FALSE); // Don't write to depth buffer
  glDisable(GL_MULTISAMPLE);
  drawBuffer(transparentData,countShader,true);
  glEnable(GL_MULTISAMPLE);
  glDepthMask(GL_TRUE); // Write to depth buffer

  if(GPUcompress) {
    compressCount();
    GLuint *p=(GLuint *) glMapBuffer(GL_ATOMIC_COUNTER_BUFFER,GL_READ_WRITE);
    elements=GPUindexing ? p[0] : p[0]-1;
    p[0]=1;
    glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER);
    if(elements == 0) return;
  } else
    elements=pixels;

  if(GPUindexing) {
    g=ceilquotient(elements,groupSize);
    elements=groupSize*g;

    if(settings::verbose > 3) {
      static bool first=true;
      if(first) {
        partialSums();
        first=false;
      }
      unsigned int N=10000;
      stopWatch Timer;
      for(unsigned int i=0; i < N; ++i)
        partialSums();
      glFinish();
      double T=Timer.seconds()/N;
      cout << "elements=" << elements << endl;
      cout << "Tmin (ms)=" << T*1e3 << endl;
      cout << "Megapixels/second=" << elements/T/1e6 << endl;
    }

    partialSums(true);
  } else {
    size_t size=elements*sizeof(GLuint);

    // Compute partial sums on the CPU
    glBindBuffer(GL_SHADER_STORAGE_BUFFER,camp::countBuffer);
    GLuint *p=(GLuint *) glMapBufferRange(GL_SHADER_STORAGE_BUFFER,
                                          0,size+sizeof(GLuint),
                                              GL_MAP_READ_BIT);
    GLuint maxsize=p[0];
    GLuint *count=p+1;

    glBindBuffer(GL_SHADER_STORAGE_BUFFER,camp::offsetBuffer);
    GLuint *offset=(GLuint *) glMapBufferRange(GL_SHADER_STORAGE_BUFFER,
                                               sizeof(GLuint),size,
                                               GL_MAP_WRITE_BIT);

    size_t Offset=offset[0]=count[0];
    for(size_t i=1; i < elements; ++i)
      offset[i]=Offset += count[i];
    fragments=Offset;

    glBindBuffer(GL_SHADER_STORAGE_BUFFER,camp::offsetBuffer);
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER,camp::countBuffer);
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

    if(exporting) {
      glBindBuffer(GL_SHADER_STORAGE_BUFFER,camp::countBuffer);
      glClearBufferData(GL_SHADER_STORAGE_BUFFER,GL_R32UI,GL_RED_INTEGER,
                        GL_UNSIGNED_INT,&zero);
    } else
      clearCount();

    if(maxsize > maxSize)
      resizeBlendShader(maxsize);
  }
  lastshader=-1;
}

void setUniforms(vertexBuffer& data, GLint shader)
{
  bool normal=shader != pixelShader;

  // Check if shader is valid
  if(shader == 0) {
    if(settings::verbose > 2) {
      cerr << "setUniforms: shader is 0!" << endl;
    }
    return;
  }

  if(shader != lastshader) {
    glUseProgram(shader);

    if(normal)
      glUniform1ui(glGetUniformLocation(shader,"width"),Width);
  }

  glUniformMatrix4fv(glGetUniformLocation(shader,"projViewMat"),1,GL_FALSE,
                     value_ptr(glm::mat4(projViewMat)));

  glUniformMatrix4fv(glGetUniformLocation(shader,"viewMat"),1,GL_FALSE,
                     value_ptr(glm::mat4(viewMat)));
  if(normal)
    glUniformMatrix3fv(glGetUniformLocation(shader,"normMat"),1,GL_FALSE,
                       value_ptr(glm::mat3(normMat)));

  if(shader == countShader) {
    lastshader=shader;
    return;
  }

  if(shader != lastshader) {
    lastshader=shader;
    glUniform1ui(glGetUniformLocation(shader,"nlights"),nlights);

    for(size_t i=0; i < nlights; ++i) {
      triple Lighti=Lights[i];
      size_t i4=4*i;
      glUniform3f(glGetUniformLocation(shader,
                                       getLightIndex(i,"direction").c_str()),
                  (GLfloat) Lighti.getx(),(GLfloat) Lighti.gety(),
                  (GLfloat) Lighti.getz());

      glUniform3f(glGetUniformLocation(shader,
                                       getLightIndex(i,"color").c_str()),
                  (GLfloat) Diffuse[i4],(GLfloat) Diffuse[i4+1],
                  (GLfloat) Diffuse[i4+2]);
    }

    // IBL textures - disabled for now due to template issues
    // if(settings::getSetting<bool>("ibl")) {
    //   camp::IBLbrdfTex.setUniform(glGetUniformLocation(shader, "reflBRDFSampler"));
    //   camp::irradianceTex.setUniform(glGetUniformLocation(shader, "diffuseSampler"));
    //   camp::reflTexturesTex.setUniform(glGetUniformLocation(shader, "reflImgSampler"));
    // }
  }

  GLuint binding=0;
  GLint blockindex=glGetUniformBlockIndex(shader,"MaterialBuffer");
  glUniformBlockBinding(shader,blockindex,binding);
  bool copy=(remesh || data.partial || !data.rendered) && !copied;
  registerBuffer(data.materials,data.materialsBuffer,copy,GL_UNIFORM_BUFFER);
  glBindBufferBase(GL_UNIFORM_BUFFER,binding,data.materialsBuffer);
}

void drawBuffer(vertexBuffer& data, GLint shader, bool color)
{
  if(data.indices.empty()) return;

  // Ensure VAO is valid (non-zero) - should already be bound from setBuffers()
  if(camp::vao == 0) {
    if(settings::verbose > 2) {
      cerr << "drawBuffer: VAO not initialized! Creating now..." << endl;
    }
    glGenVertexArrays(1, &camp::vao);
    glBindVertexArray(camp::vao);  // Bind once and leave it bound
  }

  if(settings::verbose > 2) {
    cerr << "drawBuffer: camp::vao=" << camp::vao << endl;
  }

  // Check for OpenGL errors before drawing
  GLenum err = glGetError();
  if(err != GL_NO_ERROR && settings::verbose > 2) {
    cerr << "drawBuffer: OpenGL error at start: " << err << endl;
  }

  bool normal=shader != pixelShader;

  const size_t size=sizeof(GLfloat);
  const size_t intsize=sizeof(GLint);
  const size_t bytestride=color ? sizeof(VertexData) :
    (normal ? sizeof(vertexData) : sizeof(vertexData0));

  // Debug output for material rendering
  if(settings::verbose > 2 && !data.vertices.empty()) {
    cerr << "drawBuffer: vertices.size=" << data.vertices.size()
         << " indices.size=" << data.indices.size()
         << " copy=" << ((remesh || data.partial || !data.rendered) && !copied) << endl;
  }

  // VAO is already bound from setBuffers(), no need to bind here

  bool copy=(remesh || data.partial || !data.rendered) && !copied;
  if(color) registerBuffer(data.Vertices,data.VerticesBuffer,copy);
  else if(normal) registerBuffer(data.vertices,data.verticesBuffer,copy);
  else registerBuffer(data.vertices0,data.vertices0Buffer,copy);

  registerBuffer(data.indices,data.indicesBuffer,copy,GL_ELEMENT_ARRAY_BUFFER);

  camp::setUniforms(data,shader);

  data.rendered=true;

  glVertexAttribPointer(positionAttrib,3,GL_FLOAT,GL_FALSE,bytestride,
                        (void *) 0);
  glEnableVertexAttribArray(positionAttrib);

  if(normal && Nlights > 0) {
    glVertexAttribPointer(normalAttrib,3,GL_FLOAT,GL_FALSE,bytestride,
                          (void *) (3*size));
    glEnableVertexAttribArray(normalAttrib);
  } else if(!normal) {
    glVertexAttribPointer(widthAttrib,1,GL_FLOAT,GL_FALSE,bytestride,
                          (void *) (3*size));
    glEnableVertexAttribArray(widthAttrib);
  }

  glVertexAttribIPointer(materialAttrib,1,GL_INT,bytestride,
                         (void *) ((normal ? 6 : 4)*size));
  glEnableVertexAttribArray(materialAttrib);

  if(color) {
    glVertexAttribPointer(colorAttrib,4,GL_FLOAT,GL_FALSE,bytestride,
                          (void *) (6*size+intsize));
    glEnableVertexAttribArray(colorAttrib);
  }

  fpu_trap(false); // Work around FE_INVALID
  glDrawElements(data.type,data.indices.size(),GL_UNSIGNED_INT,(void *) 0);

  // Check for OpenGL errors after draw call
  err = glGetError();
  if(err != GL_NO_ERROR && settings::verbose > 2) {
    cerr << "drawBuffer: OpenGL error after glDrawElements: " << err << endl;
  }
  fpu_trap(settings::trap());

  // Disable attribute arrays but keep VAO bound for next draw call
  glDisableVertexAttribArray(positionAttrib);
  if(normal && Nlights > 0)
    glDisableVertexAttribArray(normalAttrib);
  if(!normal)
    glDisableVertexAttribArray(widthAttrib);
  glDisableVertexAttribArray(materialAttrib);
  if(color)
    glDisableVertexAttribArray(colorAttrib);

  glBindBuffer(GL_UNIFORM_BUFFER,0);
  glBindBuffer(GL_ARRAY_BUFFER,0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,0);
  // VAO remains bound for subsequent draw operations
}

void drawMaterial0()
{
  drawBuffer(material0Data,pixelShader);
  material0Data.clear();
}

void drawMaterial1()
{
  drawBuffer(material1Data,materialShader[Opaque]);
  material1Data.clear();
}

void drawMaterial()
{
  // Check for any pending OpenGL errors before drawing
  GLenum err = glGetError();
  if(err != GL_NO_ERROR && settings::verbose > 2) {
    cerr << "drawMaterial: OpenGL error before drawBuffer: " << err << endl;
  }

  drawBuffer(materialData,materialShader[Opaque]);
  materialData.clear();
}

void drawColor()
{
  drawBuffer(colorData,colorShader[Opaque],true);
  colorData.clear();
}

void drawTriangle()
{
  drawBuffer(triangleData,generalShader[Opaque],true);
  triangleData.clear();
}

void aBufferTransparency()
{
  // Collect transparent fragments
  glDepthMask(GL_FALSE); // Disregard depth
  drawBuffer(transparentData,transparentShader,true);
  glDepthMask(GL_TRUE); // Respect depth

  // Blend transparent fragments
  glDisable(GL_DEPTH_TEST);
  glUseProgram(blendShader);
  lastshader=blendShader;
  glUniform1ui(glGetUniformLocation(blendShader,"width"),Width);
  glUniform4f(glGetUniformLocation(blendShader,"background"),
              camp::glRenderer->Background[0],camp::glRenderer->Background[1],camp::glRenderer->Background[2],
              camp::glRenderer->Background[3]);
  fpu_trap(false); // Work around FE_INVALID
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
  glDrawArrays(GL_TRIANGLES,0,3);
  fpu_trap(settings::trap());
  transparentData.clear();
  glEnable(GL_DEPTH_TEST);
}

void drawTransparent()
{
  if(camp::ssbo) {
    glDisable(GL_MULTISAMPLE);
    aBufferTransparency();
    glEnable(GL_MULTISAMPLE);
  } else {
    sortTriangles();
    transparentData.rendered=false; // Force copying of sorted triangles to GPU
    glDepthMask(GL_FALSE); // Don't write to depth buffer
    drawBuffer(transparentData,transparentShader,true);
    glDepthMask(GL_TRUE); // Write to depth buffer
    transparentData.clear();
  }
}

void drawBuffers()
{
  copied=false;
  Opaque=transparentData.indices.empty();
  bool transparent=!Opaque;

  if(settings::verbose > 2) {
    cerr << "drawBuffers: Opaque=" << Opaque
         << " material0 indices=" << material0Data.indices.size()
         << " material1 indices=" << material1Data.indices.size()
         << " material indices=" << materialData.indices.size()
         << " material vertices=" << materialData.vertices.size()
         << " color indices=" << colorData.indices.size()
         << " triangle indices=" << triangleData.indices.size()
         << " transparent indices=" << transparentData.indices.size()
         << endl;
  }

  if(camp::ssbo) {
    if(transparent) {
      refreshBuffers();
      if(!interlock) {
        resizeFragmentBuffer();
        copied=true;
      }
    }
  }

  drawMaterial0();
  drawMaterial1();
  drawMaterial();
  drawColor();
  drawTriangle();

  if(transparent) {
    if(camp::ssbo)
      copied=true;
    if(interlock) resizeFragmentBuffer();
    drawTransparent();
  }
  Opaque=0;
}

void setMaterial(vertexBuffer& data, draw_t *draw)
{
  if(materialIndex >= data.materialTable.size() ||
     data.materialTable[materialIndex] == -1) {
    if(data.materials.size() >= Maxmaterials) {
      data.partial=true;
      (*draw)();
    }
    size_t size0=data.materialTable.size();
    data.materialTable.resize(materialIndex+1);
    for(size_t i=size0; i < materialIndex; ++i)
      data.materialTable[i]=-1;
    data.materialTable[materialIndex]=data.materials.size();
    data.materials.push_back(materials[materialIndex]);
  }
  materialIndex=data.materialTable[materialIndex];
}

}
#endif /* HAVE_RENDERER */

#ifdef HAVE_RENDERER
namespace camp {

AsyGLRender::~AsyGLRender()
{
#ifdef HAVE_RENDERER
  if (this->View && glfwWindow != nullptr) {
    ::glfwDestroyWindow(static_cast<GLFWwindow*>(glfwWindow));
    glfwWindow = nullptr;
  }

  // Cleanup OpenGL resources
  glDeleteProgram(pixelShader);
  for(int i=0; i<2; ++i) {
    glDeleteProgram(materialShader[i]);
    glDeleteProgram(colorShader[i]);
    glDeleteProgram(generalShader[i]);
  }
  glDeleteProgram(countShader);
  glDeleteProgram(transparentShader);
  glDeleteProgram(blendShader);
  glDeleteProgram(zeroShader);
  glDeleteProgram(compressShader);
  glDeleteProgram(sum1Shader);
  glDeleteProgram(sum2Shader);
  glDeleteProgram(sum2fastShader);
  glDeleteProgram(sum3Shader);

  if(vao) glDeleteVertexArrays(1, &vao);
  if(offsetBuffer) glDeleteBuffers(1, &offsetBuffer);
  if(indexBuffer) glDeleteBuffers(1, &indexBuffer);
  if(elementsBuffer) glDeleteBuffers(1, &elementsBuffer);
  if(countBuffer) glDeleteBuffers(1, &countBuffer);
  if(globalSumBuffer) glDeleteBuffers(1, &globalSumBuffer);
  if(fragmentBuffer) glDeleteBuffers(1, &fragmentBuffer);
  if(depthBuffer) glDeleteBuffers(1, &depthBuffer);
  if(opaqueBuffer) glDeleteBuffers(1, &opaqueBuffer);
  if(opaqueDepthBuffer) glDeleteBuffers(1, &opaqueDepthBuffer);
  if(feedbackBuffer) glDeleteBuffers(1, &feedbackBuffer);
#endif
}

void AsyGLRender::render(RenderFunctionArgs const& args)
{
  // Initialize GLFW and get screen dimensions FIRST (matching reference pattern)
#ifdef HAVE_LIBGLFW
#ifndef HAVE_LIBOSMESA
  static bool glfwInitialized = false;
  if(!glfwInitialized) {
    glfwSetErrorCallback([](int error, const char* description) {
      cerr << "GLFW error [" << error << "]: " << description << endl;
    });

    if(!::glfwInit()) {
      cerr << "Failed to initialize GLFW" << endl;
      exit(-1);
    }
    glfwInitialized = true;

    // Get monitor based on device setting (same as reference)
    Int device = getSetting<Int>("device");
    int numMonitors;
    GLFWmonitor** monitors = glfwGetMonitors(&numMonitors);

    GLFWmonitor* monitor = nullptr;
    if (monitors && numMonitors > 0) {
      int monitorIndex = (int)device;
      if (monitorIndex < 0) monitorIndex = numMonitors + monitorIndex;
      if (monitorIndex >= 0 && monitorIndex < numMonitors)
        monitor = monitors[monitorIndex];
      else
        monitor = glfwGetPrimaryMonitor();
    } else {
      monitor = glfwGetPrimaryMonitor();
    }

    if(monitor) {
      int mx, my;
      glfwGetMonitorWorkarea(monitor, &mx, &my, &screenWidth, &screenHeight);
    } else {
      // Fallback if no monitor found (e.g., no X display)
      screenWidth = maxTileWidth > 0 ? maxTileWidth : 1024;
      screenHeight = maxTileHeight > 0 ? maxTileHeight : 768;
    }
  }
#endif
#endif

  // Initialize from arguments (following Vulkan pattern - use member variables)
  Prefix = args.prefix;
  Picture = args.pic;  // Set global Picture variable for drawscene()
  pic = args.pic;      // Also set member variable
  Format = args.format;
  nlights = args.nlightsin;

  Lights = args.lights;
  Diffuse = args.diffuse;
  Specular = args.specular;
  View = args.view;
  Angle = args.angle * ASY_RADIANS;
  Zoom0 = std::fpclassify(args.zoom) == FP_NORMAL ? args.zoom : 1.0;
  Shift = args.shift / Zoom0;
  Margin = args.margin;

  for(int i=0; i<4; ++i) {
    Background[i] = static_cast<float>(args.background[i]);
  }

  // Use member variables from AsyRender base class (following Vulkan pattern)
  Xmin = args.m.getx();
  Xmax = args.M.getx();
  Ymin = args.m.gety();
  Ymax = args.M.gety();
  Zmin = args.m.getz();
  Zmax = args.M.getz();

  // Also set lowercase viewport bounds (member variables from AsyRender)
  xmin = Xmin;
  xmax = Xmax;
  ymin = Ymin;
  ymax = Ymax;

  cerr << "DEBUG: After setting - Xmin=" << Xmin << " Ymin=" << Ymin << " Zmin=" << Zmin << endl;
  cerr << "DEBUG: After setting - Xmax=" << Xmax << " Ymax=" << Ymax << " Zmax=" << Zmax << endl;
  cerr << "DEBUG: After setting - xmin=" << xmin << " xmax=" << xmax << " ymin=" << ymin << " ymax=" << ymax << endl;

  haveScene = Xmin < Xmax && Ymin < Ymax && Zmin < Zmax;
  orthographic = Angle == 0.0;
  H = orthographic ? 0.0 : -tan(0.5 * Angle) * Zmax;
  Xfactor = Yfactor = 1.0;

  // Sync minimal globals with member variables (following Vulkan pattern)
  camp::orthographic_gl = orthographic;

  for(int i=0; i<16; ++i) {
    T[i] = args.t[i];
    Tup[i] = args.tup[i];
  }

  // Initialize window dimensions and aspect ratio
  bool v3d = args.format == "v3d";
  bool webgl = args.format == "html";
  bool format3d = webgl || v3d;

  antialias = settings::getSetting<Int>("antialias") > 1;
  double expand;
  if(format3d) {
    expand = 1.0;
  } else {
    expand = settings::getSetting<double>("render");
    if(expand < 0)
      expand *= (Format.empty() || Format == "eps" || Format == "pdf") ? -2.0 : -1.0;
    if(antialias) expand *= 2.0;
  }

  oWidth = args.width;
  oHeight = args.height;
  Aspect = args.width / args.height;
  Aspect = args.width / args.height;

  pair maxViewport = settings::getSetting<pair>("maxviewport");
  int maxWidth = maxViewport.getx() > 0 ? (int)ceil(maxViewport.getx()) : screenWidth;
  int maxHeight = maxViewport.gety() > 0 ? (int)ceil(maxViewport.gety()) : screenHeight;
  if(maxWidth <= 0) maxWidth = max(maxHeight, 2);
  if(maxHeight <= 0) maxHeight = max(maxWidth, 2);

  if(screenWidth <= 0) screenWidth = maxWidth;
  else screenWidth = min(screenWidth, maxWidth);
  if(screenHeight <= 0) screenHeight = maxHeight;
  else screenHeight = min(screenHeight, maxHeight);

  fullWidth = (int)ceil(expand * args.width);
  fullHeight = (int)ceil(expand * args.height);

  if(format3d) {
    Width = fullWidth;
    Height = fullHeight;
  } else {
    Width = min(fullWidth, screenWidth);
    Height = min(fullHeight, screenHeight);
    if(Width > Height * Aspect)
      Width = min((int)ceil(Height * Aspect), screenWidth);
    else
      Height = min((int)ceil(Width / Aspect), screenHeight);
  }

  // Initialize view state
  home(format3d);
  setProjection();

  if(format3d) {
    remesh = true;
    return;
  }

  maxFragments = 0;
  ArcballFactor = 1 + 8.0 * hypot(Margin.getx(), Margin.gety()) / hypot(Width, Height);
  Aspect = (double)Width / Height;

#ifdef HAVE_LIBGLFW
  setosize();
#endif

  // Create GLFW window BEFORE OpenGL initialization if viewing and not using OSMesa
#ifdef HAVE_LIBGLFW
#ifndef HAVE_LIBOSMESA
  if(View && glfwWindow == nullptr) {
    // Use appropriate window size - for hidden windows use maxTile dimensions
    int winWidth = Width;
    int winHeight = Height;
    if(!View || Iconify) {
      // For hidden/offscreen rendering, use larger tile dimensions
      winWidth = maxTileWidth > 0 ? maxTileWidth : 1024;
      winHeight = maxTileHeight > 0 ? maxTileHeight : 768;
    }

    GLFWwindow* newWindow = glfwCreateRenderWindow(winWidth, winHeight, title.empty() ? Prefix.c_str() : title.c_str(), this);
    if(newWindow == nullptr) {
      cerr << "Failed to create GLFW window" << endl;
      exit(-1);
    }
    glfwWindow = static_cast<void*>(newWindow);
    // Also set the global window variable for compatibility with existing code
    ::window = newWindow;

    // Make context current before GLEW initialization (matching reference pattern)
    glfwMakeContextCurrent(::window);

    // Initialize GLEW immediately after context creation (matching reference pattern)
    glewExperimental = GL_TRUE;
    GLenum glewErr = glewInit();
    if(glewErr != GLEW_OK) {
      cerr << "GLEW initialization error: " << glewGetErrorString(glewErr) << endl;
      exit(-1);
    }

    // Set GLSL version immediately after GLEW init (matching reference pattern)
    const char *GLSL_VERSION=(const char *)glGetString(GL_SHADING_LANGUAGE_VERSION);
    if(GLSL_VERSION)
      GLSLversion=(int) (100*atof(GLSL_VERSION)+0.5);
    if(settings::verbose > 2)
      cout << "GLSL version " << GLSL_VERSION << " (GLSLversion=" << GLSLversion << ")" << endl;

    if(settings::verbose > 2) {
      cerr << "Created window and initialized GLEW: " << Width << "x" << Height
           << " window=" << ::window << endl;
    }
  }
#endif
#endif

  // Initialize GPU settings - compute shaders disabled by default for stability
  // Only enable if explicitly set AND we have a valid OpenGL context
#if defined(HAVE_COMPUTE_SHADER) && !defined(HAVE_LIBOSMESA)
  bool gpuIndexingRequested = settings::getSetting<bool>("GPUindexing");
  GPUcompress = settings::getSetting<bool>("GPUcompress");
  // Only enable GPUindexing if explicitly requested AND we have a valid window/context
#ifdef HAVE_LIBGLFW
  if(gpuIndexingRequested && ::window) {
    if(glfwGetCurrentContext() == ::window) {
      GPUindexing = true;
    } else {
      GPUindexing = false;
      if(settings::verbose > 2)
        cout << "No valid OpenGL context for compute shaders" << endl;
    }
  } else {
    GPUindexing = false;
  }
#else
  GPUindexing = gpuIndexingRequested;
#endif
#else
  GPUindexing = false;
  GPUcompress = false;
#endif

  // Initialize OpenGL if needed
  if(initialize) {
    initialize = false;

#ifdef HAVE_LIBGLFW
#ifndef HAVE_LIBOSMESA
    // GLEW already initialized right after window creation above
    // Just verify the context is still current
    if(::window) {
      glfwMakeContextCurrent(::window);
      GLFWwindow* current = glfwGetCurrentContext();
      if(settings::verbose > 2) {
        cerr << "Post-window GLEW check: ::window=" << ::window << " current=" << current << endl;
      }
      if(current != ::window) {
        cerr << "Failed to make OpenGL context current" << endl;
        exit(-1);
      }
    } else {
      cerr << "No OpenGL window/context available" << endl;
      exit(-1);
    }
#endif
#endif

    // Check for any OpenGL errors after GLEW init (done earlier)
    GLenum glerr = glGetError();
    if(glerr != GL_NO_ERROR) {
      cerr << "OpenGL error after GLEW init: " << glerr << endl;
    }
    const char* gl_version_str = (const char*)glGetString(GL_VERSION);
    if(gl_version_str) {
      int major = gl_version_str[0] - '0';
      int minor = gl_version_str[2] - '0';
      if(major < 3 || (major == 3 && minor < 0)) {
        cerr << "OpenGL version too low: " << gl_version_str
             << " (need at least 3.0)" << endl;
        exit(-1);
      }
    }

    const char* GLSL_VERSION = (const char*)glGetString(GL_SHADING_LANGUAGE_VERSION);
    if(GLSL_VERSION) {
      GLSLversion = (int)(100 * atof(GLSL_VERSION) + 0.5);
      if(GLSLversion < 130) {
        cerr << "Unsupported GLSL version: " << GLSL_VERSION << endl;
        exit(-1);
      }
    }

    // Check multisampling
    int samples = 0;
    glGetIntegerv(GL_SAMPLES, &samples);
    if(settings::verbose > 1 && samples > 1) {
      cout << "Multisampling enabled with sample width " << samples << endl;
    }

    ibl = settings::getSetting<bool>("ibl");
    initShaders();
    setBuffers();

    // Initialize GPU compute parameters
    if(GPUindexing) {
      localSize = settings::getSetting<Int>("GPUlocalSize");
      blockSize = settings::getSetting<Int>("GPUblockSize");
      groupSize = localSize * blockSize;
    }
  }

  GLint val;
  glGetIntegerv(GL_MAX_UNIFORM_BLOCK_SIZE, &val);
  Maxmaterials = val / sizeof(Material);
  if(nmaterials > Maxmaterials) nmaterials = Maxmaterials;

  glClearColor(args.background[0], args.background[1],
               args.background[2], args.background[3]);

#ifndef HAVE_LIBOSMESA
  if(View) {
    if(!getSetting<bool>("fitscreen"))
      Fitscreen=0;
    firstFit=true;
    fitscreen();
    setosize();
  }
#endif

  glEnable(GL_DEPTH_TEST);
  if(!ssbo) {
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  }

  Mode = 2;
  cycleMode();

  ViewExport = View;
#ifdef HAVE_LIBOSMESA
  View = false;
#endif

  // Enter main loop or export
  if(View) {
    // Ensure initial render happens (matching reference pattern)
    redraw = true;
    mainLoop();
  } else {
    update();
    display();
    if(renderThread) {
      exportHandler();
    } else {
      exportHandler();
      quit();
    }
  }
}

// RenderCallbacks interface implementation
void AsyGLRender::onMouseButton(int button, int action, int mods)
{
    auto const currentActionStr = getGLFWAction(button, mods);
    if (currentActionStr.empty()) return;
    if (action == GLFW_PRESS) {
        lastAction = currentActionStr;
        // Capture initial position for movement tracking
        double xpos, ypos;
        if(glfwWindow) {
            glfwGetCursorPos(static_cast<GLFWwindow*>(glfwWindow), &xpos, &ypos);
            xprev = xpos;
            yprev = ypos;
        }
    } else if (action == GLFW_RELEASE) {
        lastAction.clear();
    }
}

void AsyGLRender::onFramebufferResize(int width, int height)
{
    if(width == 0 || height == 0) return;
    if(width == Width && height == Height) return;
    reshape0(width, height);
    update();
    remesh = true;
}

void AsyGLRender::onScroll(double xoffset, double yoffset)
{
    auto zoomFactor = getSetting<double>("zoomfactor");
    if(zoomFactor > 0.0) {
        if (yoffset > 0) camp::glRenderer->Zoom *= zoomFactor;
        else camp::glRenderer->Zoom /= zoomFactor;
    }
    capzoom();
    setProjection();
    redraw = true;
}

void AsyGLRender::onCursorPos(double xpos, double ypos)
{
    if (lastAction == "rotate") {
        camp::Arcball arcball(xprev * 2 / Width - 1, 1 - yprev * 2 / Height,
                        xpos * 2 / Width - 1, 1 - ypos * 2 / Height);
        camp::triple axis = arcball.axis;
        rotateMat = glm::rotate(2 * arcball.angle / camp::glRenderer->Zoom * ArcballFactor,
                           glm::dvec3(axis.getx(), axis.gety(), axis.getz())) * rotateMat;
        update();
    } else if (lastAction == "shift") {
        shift(xpos - xprev, ypos - yprev);
        update();
    } else if (lastAction == "pan") {
        if (orthographic_gl) shift(xpos - xprev, ypos - yprev);
        else pan(xpos - xprev, ypos - yprev);
        update();
    } else if (lastAction == "zoom") {
        zoom(0.0, ypos - yprev);
    }
    xprev = xpos;
    yprev = ypos;
}

void AsyGLRender::onKey(int key, int scancode, int action, int mods)
{
    AsyRender::onKey(key, scancode, action, mods);
}

void AsyGLRender::onWindowFocus(int focused) {}

void AsyGLRender::onClose()
{
    AsyRender::onClose();
    exitHandler(0);
}

void AsyGLRender::display()
{
#ifdef HAVE_RENDERER
  GLFWwindow* win = static_cast<GLFWwindow*>(glfwWindow);
  if(View && glfwWindow != nullptr) {
    // Make OpenGL context current before any GL operations
    ::glfwMakeContextCurrent(win);

    if(!hideWindow && !glfwGetWindowAttrib(win,GLFW_VISIBLE))
      ::glfwShowWindow(win);
  }
#endif

  drawscene(Width, Height);

  bool fps=settings::verbose > 2;
  if(fps) {
    if(framecount < 20) fpsTimer.reset();
    else {
      double s=fpsTimer.seconds(true);
      if(s > 0.0) {
        double rate=1.0/s;
        fpsStats.add(rate);
        if(framecount % 20 == 0)
          cout << "FPS=" << rate << "\t" << fpsStats.mean()
               << " +/- " << fpsStats.stdev() << endl;
      }
    }
    ++framecount;
  }

#ifdef HAVE_RENDERER
  if(glfwWindow) glfwSwapBuffers(static_cast<GLFWwindow*>(glfwWindow));
#endif

  if(!renderThread) {
#if defined(_WIN32)
#else
    // Oldpid is now in AsyRender base class
    if(Oldpid != 0 && waitpid(Oldpid,NULL,WNOHANG) != Oldpid) {
      kill(Oldpid,SIGHUP);
      Oldpid=0;
    }
#endif
  }
}

void AsyGLRender::setProjection()
{
  AsyRender::setProjection();
}

void AsyGLRender::updateModelViewData()
{
  normMat=dmat3(glm::inverse(viewMat));
  const double *T=value_ptr(normMat);
  for(size_t i=0; i < 9; ++i)
    glBBT[i]=T[i];
}

/*
void AsyGLRender::update()
{
  if(glfwWindow) ::glfwShowWindow(static_cast<GLFWwindow*>(glfwWindow));
  AsyRender::update();
  }*/

void AsyGLRender::mainLoop()
{
#ifdef HAVE_RENDERER
  if(View && glfwWindow != nullptr) {
    GLFWwindow* win = static_cast<GLFWwindow*>(glfwWindow);

    glfwRunLoop(win,
      [win](){
        if (win == nullptr) return false;
        return !glfwWindowShouldClose(win);
      },
      [this](){ return redraw || redisplay || queueExport; },
      [this](){
        redisplay=false;
        redraw=false;
        waitEvent=true;
        if(resize) { fitscreen(!interact::interactive); resize=false; }
        display();
      },
      nullptr,
      [this](){ return currentIdleFunc; },
      [this](){ return waitEvent; }
    );
  } else {
    update();
    display();
    if(renderThread) exportHandler();
    else { exportHandler(); quit(); }
  }
#endif
}

void AsyGLRender::exportHandler(int)
{
#ifdef HAVE_RENDERER
  readyAfterExport=true;
#endif
  Export();
}

void AsyGLRender::reshape0(int width, int height)
{
  AsyRender::reshape0(width, height);
}

} // namespace camp

#endif // HAVE_LIBGLM

#endif // HAVE_RENDERER
