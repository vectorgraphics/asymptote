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
#include "exithandlers.h"

#include "picture.h"
#include "bbox3.h"
#include "drawimage.h"
#include "interact.h"
#include "fpu.h"
#include "renderBase.h"

extern uint32_t CLZ(uint32_t a);

// GPU settings - class members now, but kept as globals for shader compilation params
bool GPUindexing = false;  // Disabled by default - compute shaders not needed for opaque rendering
bool GPUcompress;

#ifdef HAVE_RENDERER
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
// Global matrices for shader compatibility (accessed from setUniforms)
glm::mat3 normMat;   // Normal matrix is 3x3, not 4x4
double glBBT[9] = {0};
const double *dView;

Billboard BB;

// Vertex buffers - these remain globals as they are populated by drawElement rendering
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

bool orthographic_gl = false;

void clearCenters()
{
  drawElement::centers.clear();
  drawElement::centermap.clear();
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

// Note: different name to avoid conflict with v3dheadertypes::orthographic enum
extern double Angle, Zoom0;
extern pair Shift, Margin;
extern double T[16], Tup[16];
extern double Xmin, Xmax, Ymin, Ymax, Zmin, Zmax;  // These are now member variables in AsyRender

bool Xspin,Yspin,Zspin;

double T[16];
double Tup[16];

// OpenGL-specific global state (minimal set for shader compatibility)
bool Iconify=false;
bool ignorezoom;
int Fitscreen=1;
bool firstFit;
bool ViewExport;
int Oldpid;
// Note: Prefix, Format, fullWidth, fullHeight, Width, Height are now in AsyRender base class
const picture* Picture;  // Keep as global for drawscene() compatibility
double gl_X, gl_Y;
string currentAction="";
double cx,cy;
double xprev,yprev;
static const double ASY_PI=acos(-1.0);
static const double ASY_DEGREES=180.0/ASY_PI;
static const double ASY_RADIANS=1.0/ASY_DEGREES;

bool format3dWait=false;  // Keep as global for threading

// IBL textures - disabled for now due to template issues
void* iblbrdfTex = nullptr;
void* irradianceTex = nullptr;
void* reflTexturesTex = nullptr;

glm::vec4 vec4(triple v)
{
  return glm::vec4(v.getx(),v.gety(),v.getz(),0);
}

glm::vec4 vec4(double *v)
{
  return glm::vec4(v[0],v[1],v[2],v[3]);
}

#ifdef HAVE_RENDERER

// GLFW window globals - kept in camp namespace for type compatibility
#ifdef HAVE_LIBGLFW
int oldWidth,oldHeight;

bool queueScreen=false;

string Action;

double lastangle;
#endif

using utils::statistics;
statistics S;

camp::GLTexture2<float,GL_FLOAT> fromEXR(string const& EXRFile, camp::GLTexturesFmt const& fmt, GLint const& textureNumber)
{
  IEXRFile fil(EXRFile);
  return camp::GLTexture2<float,GL_FLOAT> {fil.getData(),fil.size(),textureNumber,fmt};
}

camp::GLTexture3<float,GL_FLOAT> fromEXR3(
  mem::vector<string> const& EXRFiles, camp::GLTexturesFmt const& fmt, GLint const& textureNumber)
{
  // 3d reflectance textures
  std::vector<float> data;
  size_t count=EXRFiles.size();
  int wi=0, ht=0;

  for(string const& EXRFile : EXRFiles) {
    IEXRFile fil3(EXRFile);
    std::tie(wi,ht)=fil3.size();
    size_t imSize=4*wi*ht;
    std::copy(fil3.getData(),fil3.getData()+imSize,std::back_inserter(data));
  }

  return camp::GLTexture3<float,GL_FLOAT> {
          data.data(),
          std::tuple<int,int,int>(wi,ht,static_cast<int>(count)),textureNumber,
          fmt
  };
}

void initIBL()
{
  camp::GLTexturesFmt fmt;
  fmt.internalFmt=GL_RGB16F;
  string imageDir=locateFile(getSetting<string>("imageDir"))+"/";
  string imagePath=imageDir+getSetting<string>("image")+"/";
  // IBL textures - disabled for now due to template issues
  // irradianceTex=fromEXR(imagePath+"diffuse.exr",fmt,1);
  // camp::GLTexturesFmt fmtRefl;
  // fmtRefl.internalFmt=GL_RG16F;
  // IBLbrdfTex=fromEXR(imageDir+"refl.exr",fmtRefl,2);

  camp::GLTexturesFmt fmt3;
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
  // reflTexturesTex=fromEXR3(files,fmt3,3);
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
  s << "LOCALSIZE " << gl->localSize << "u" << endl;
  shaderParams.push_back(s.str().c_str());
  s2 << "BLOCKSIZE " << gl->blockSize << "u" << endl;
  shaderParams.push_back(s2.str().c_str());
  GLuint rc=compileAndLinkShader(shaders,shaderParams,true,false,true,true);
  if(rc == 0) {
    GPUindexing=false; // Compute shaders are unavailable.
    if(settings::verbose > 2)
      cout << "No compute shader support" << endl;
  } else {
//    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT,0,&maxgroups);
//    maxgroups=min(1024,maxgroups/(GLint) (gl->localSize*gl->localSize));
    gl->sum1Shader=rc;

    shaders[0]=ShaderfileModePair(sum2.c_str(),GL_COMPUTE_SHADER);
    gl->sum2Shader=compileAndLinkShader(shaders,shaderParams,true,false,
                                          true);

    shaders[0]=ShaderfileModePair(sum2fast.c_str(),GL_COMPUTE_SHADER);
    gl->sum2fastShader=compileAndLinkShader(shaders,shaderParams,true,false,
                                              true);

    shaders[0]=ShaderfileModePair(sum3.c_str(),GL_COMPUTE_SHADER);
    gl->sum3Shader=compileAndLinkShader(shaders,shaderParams,true,false,
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
  s << "ARRAYSIZE " << gl->maxSize << "u" << endl;
  shaderParams.push_back(s.str().c_str());
  if(GPUindexing)
    shaderParams.push_back("GPUINDEXING");
  if(GPUcompress)
    shaderParams.push_back("GPUCOMPRESS");
  shaders[0]=ShaderfileModePair(screen.c_str(),GL_VERTEX_SHADER);
  shaders[1]=ShaderfileModePair(blend.c_str(),GL_FRAGMENT_SHADER);
  gl->blendShader=compileAndLinkShader(shaders,shaderParams,gl->ssbo);
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

void setBuffers()
{
  if(settings::verbose > 2) {
    cerr << "setBuffers: Creating VAO, gl->vao=" << gl->vao << endl;
  }
  glGenVertexArrays(1,&gl->vao);
  if(settings::verbose > 2) {
    cerr << "setBuffers: VAO created, gl->vao=" << gl->vao << endl;
  }
  // Bind VAO once and leave it bound for all subsequent draw operations
  glBindVertexArray(gl->vao);

  material0Data.reserve0();
  materialData.reserve();
  colorData.Reserve();
  triangleData.Reserve();
  transparentData.Reserve();

#ifdef HAVE_SSBO
  glGenBuffers(1, &gl->offsetBuffer);
  if(GPUindexing)
    glGenBuffers(1, &gl->globalSumBuffer);
  glGenBuffers(1, &gl->feedbackBuffer);
  glGenBuffers(1, &gl->countBuffer);
  if(GPUcompress) {
    glGenBuffers(1, &gl->indexBuffer);
    glGenBuffers(1, &gl->elementsBuffer);
  }
  glGenBuffers(1, &gl->fragmentBuffer);
  glGenBuffers(1, &gl->depthBuffer);
  glGenBuffers(1, &gl->opaqueBuffer);
  glGenBuffers(1, &gl->opaqueDepthBuffer);
#endif

  if(settings::verbose > 2) {
    cerr << "setBuffers: Done, gl->vao=" << gl->vao << endl;
  }
}

void initShaders()
{
  gl->Nlights = gl->nlights == 0 ? 0 : std::max(gl->Nlights, gl->nlights);
  Nmaterials = std::max(Nmaterials, nmaterials);

  string zero=locateFile("shaders/zero.glsl");
  string compress=locateFile("shaders/compress.glsl");
  string vertex=locateFile("shaders/vertex.glsl");
  string count=locateFile("shaders/count.glsl");
  string fragment=locateFile("shaders/fragment.glsl");
  string screen=locateFile("shaders/screen.glsl");

  if(zero.empty() || compress.empty() || vertex.empty() || fragment.empty() ||
     screen.empty() || count.empty())
    noShaders();

  // Only try compute shaders if GPUindexing is explicitly enabled
  if(GPUindexing) {
    initComputeShaders();
  }

  std::vector<ShaderfileModePair> shaders(2);
  std::vector<std::string> shaderParams;

  if(gl->ibl) {
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
  gl->countShader=compileAndLinkShader(shaders,shaderParams,
                                         true,false,false,true);
  if(gl->countShader)
    shaderParams.push_back("HAVE_SSBO");
#else
  gl->countShader=0;
#endif

  gl->ssbo=gl->countShader;
#ifdef HAVE_LIBOSMESA
  gl->interlock=false;
#else
  gl->interlock=gl->ssbo && getSetting<bool>("GPUinterlock");
#endif

  if(!gl->ssbo && settings::verbose > 2)
    cout << "No SSBO support; order-independent transparency unavailable"
         << endl;

  shaders[1]=ShaderfileModePair(fragment.c_str(),GL_FRAGMENT_SHADER);
  shaderParams.push_back("MATERIAL");
  if(orthographic_gl)
    shaderParams.push_back("ORTHOGRAPHIC");

  ostringstream lights,materials,opaque;
  lights << "Nlights " << gl->Nlights;
  shaderParams.push_back(lights.str().c_str());
  materials << "Nmaterials " << Nmaterials;
  shaderParams.push_back(materials.str().c_str());

  shaderParams.push_back("WIDTH");
  gl->pixelShader=compileAndLinkShader(shaders,shaderParams,gl->ssbo);
  shaderParams.pop_back();

  shaderParams.push_back("NORMAL");
  if(gl->interlock) shaderParams.push_back("HAVE_INTERLOCK");
  gl->materialShader[0]=compileAndLinkShader(shaders,shaderParams,
                                               gl->ssbo,gl->interlock,false,true);
  if(gl->interlock && !gl->materialShader[0]) {
    shaderParams.pop_back();
    gl->interlock=false;
    gl->materialShader[0]=compileAndLinkShader(shaders,shaderParams,gl->ssbo);
    if(settings::verbose > 2)
      cout << "No fragment shader interlock support" << endl;
  }

  shaderParams.push_back("OPAQUE");
  gl->materialShader[1]=compileAndLinkShader(shaders,shaderParams,gl->ssbo);
  shaderParams.pop_back();

  shaderParams.push_back("COLOR");
  gl->colorShader[0]=compileAndLinkShader(shaders,shaderParams,gl->ssbo,
                                            gl->interlock);
  shaderParams.push_back("OPAQUE");
  gl->colorShader[1]=compileAndLinkShader(shaders,shaderParams,gl->ssbo);
  shaderParams.pop_back();

  shaderParams.push_back("GENERAL");
  if(gl->mode != DRAWMODE_NORMAL)
    shaderParams.push_back("WIREFRAME");
  gl->generalShader[0]=compileAndLinkShader(shaders,shaderParams,gl->ssbo,
                                              gl->interlock);
  shaderParams.push_back("OPAQUE");
  gl->generalShader[1]=compileAndLinkShader(shaders,shaderParams,gl->ssbo);
  shaderParams.pop_back();

  shaderParams.push_back("TRANSPARENT");
  gl->transparentShader=compileAndLinkShader(shaders,shaderParams,gl->ssbo,
                                               gl->interlock);
  shaderParams.clear();

  if(gl->ssbo) {
    if(GPUindexing)
      shaderParams.push_back("GPUINDEXING");
    shaders[0]=ShaderfileModePair(screen.c_str(),GL_VERTEX_SHADER);
    shaders[1]=ShaderfileModePair(compress.c_str(),GL_FRAGMENT_SHADER);
    gl->compressShader=compileAndLinkShader(shaders,shaderParams,gl->ssbo);
    if(GPUindexing)
      shaderParams.pop_back();
    else {
      shaders[1]=ShaderfileModePair(zero.c_str(),GL_FRAGMENT_SHADER);
      gl->zeroShader=compileAndLinkShader(shaders,shaderParams,gl->ssbo);
    }
    gl->maxSize=1;
    initBlendShader();
  }
  gl->lastshader=-1;

  if(gl->vao == 0)
    setBuffers();
}

void deleteComputeShaders()
{
  glDeleteProgram(gl->sum1Shader);
  glDeleteProgram(gl->sum2Shader);
  glDeleteProgram(gl->sum2fastShader);
  glDeleteProgram(gl->sum3Shader);
}

void deleteBlendShader()
{
  glDeleteProgram(gl->blendShader);
}

void deleteShaders()
{
  if(gl->ssbo) {
    deleteBlendShader();
    if(GPUindexing)
      deleteComputeShaders();
    else
      glDeleteProgram(gl->zeroShader);
    glDeleteProgram(gl->countShader);
    glDeleteProgram(gl->compressShader);
  }

  if (gl->transparentShader != 0)
    glDeleteProgram(gl->transparentShader);
  for(unsigned int opaque=0; opaque < 2; ++opaque) {
    if (gl->generalShader[opaque] != 0)
      glDeleteProgram(gl->generalShader[opaque]);
    if (gl->colorShader[opaque] != 0)
      glDeleteProgram(gl->colorShader[opaque]);
    if (gl->materialShader[opaque] != 0)
      glDeleteProgram(gl->materialShader[opaque]);
  }
  if (gl->pixelShader != 0)
    glDeleteProgram(gl->pixelShader);
}

void resizeBlendShader(GLuint maxsize)
{
  gl->maxSize=ceilpow2(maxsize);
  deleteBlendShader();
  initBlendShader();
}

bool exporting=false;

void drawscene(int Width, int Height)
{
#ifdef HAVE_PTHREAD
  static bool first=true;
  if(first) {
    gl->wait(gl->initSignal,gl->initLock);
    gl->endwait(gl->initSignal,gl->initLock);
    first=false;
  }

  if(format3dWait)
    gl->wait(gl->initSignal,gl->initLock);
#endif

  if((gl->nlights == 0 && gl->Nlights > 0) || gl->nlights > gl->Nlights ||
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
  if(gl->xmin >= gl->xmax || gl->ymin >= gl->ymax || gl->Zmin >= gl->Zmax) return;

  triple m(gl->xmin,gl->ymin,gl->Zmin);
  triple M(gl->xmax,gl->ymax,gl->Zmax);
  double perspective=gl->orthographic || gl->Zmax == 0.0 ? 0.0 : 1.0/gl->Zmax;

  double size2=hypot(Width,Height);

  if(gl->remesh)
    clearCenters();

  if(settings::verbose > 2) {
    cerr << "drawscene: calling Picture->render()" << endl;
  }
  if(Picture)
    Picture->render(size2,m,M,perspective,gl->remesh);

  if(settings::verbose > 2) {
    cerr << "drawscene: Picture->render() complete" << endl;
  }

#ifdef HAVE_RENDERER
  drawBuffers();
#endif

  if(gl->outlinemode) gl->remesh=false;
}

// Return x divided by y rounded up to the nearest integer.
int ceilquotient(int x, int y)
{
  return (x+y-1)/y;
}

void Export()
{
  size_t ndata=3*gl->fullWidth*gl->fullHeight;
  if(ndata == 0) return;
  glReadBuffer(GL_BACK_LEFT);
  glPixelStorei(GL_PACK_ALIGNMENT,1);
  glFinish();
  exporting=true;

  try {
    unsigned char *data=new unsigned char[ndata];
    if(data) {
      TRcontext *tr=trNew();
      int width=ceilquotient(gl->fullWidth,
                             ceilquotient(gl->fullWidth,std::min(gl->maxTileWidth,gl->Width)));
      int height=ceilquotient(gl->fullHeight,
                              ceilquotient(gl->fullHeight,
                                           std::min(gl->maxTileHeight,gl->Height)));
      if(settings::verbose > 1)
        cout << "Exporting " << gl->Prefix << " as " << gl->fullWidth << "x"
             << gl->fullHeight << " image" << " using tiles of size "
             << width << "x" << height << endl;

      unsigned border=std::min(std::min(1,(width-1)/2),(height-1)/2);
      trTileSize(tr,width,height,border);
      trImageSize(tr,gl->fullWidth,gl->fullHeight);
      trImageBuffer(tr,GL_RGB,GL_UNSIGNED_BYTE,data);

      // Use member variables from AsyGLRender (following Vulkan pattern)
      double dXmin = gl->xmin;
      double dXmax = gl->xmax;
      double dYmin = gl->ymin;
      double dYmax = gl->ymax;
      double dZmin = gl->Zmin;
      double dZmax = gl->Zmax;

      size_t count=0;
      if(gl->haveScene) {
        (orthographic_gl ? trOrtho : trFrustum)(tr,dXmin,dXmax,dYmin,dYmax,-dZmax,-dZmin);
        do {
          trBeginTile(tr);
          gl->remesh=true;
          drawscene(gl->fullWidth,gl->fullHeight);
          gl->lastshader=-1;
          ++count;
        } while (trEndTile(tr));
      } else {// clear screen and return
        drawscene(gl->fullWidth,gl->fullHeight);
      }

      if(settings::verbose > 1)
        cout << count << " tile" << (count != 1 ? "s" : "") << " drawn" << endl;
      trDelete(tr);

      picture pic;
      drawRawImage *Image=NULL;
      if(gl->haveScene) {
        double w=gl->oWidth;
        double h=gl->oHeight;
        double Aspect=((double) gl->fullWidth)/gl->fullHeight;
        if(w > h*Aspect) w=(int) (h*Aspect+0.5);
        else h=(int) (w/Aspect+0.5);
        // Render an antialiased image.

        Image=new drawRawImage(data,gl->fullWidth,gl->fullHeight,
                               transform(0.0,0.0,w,0.0,0.0,h),
                               gl->antialias);
        pic.append(Image);
      }

      pic.shipout(NULL,gl->Prefix,gl->Format,false,ViewExport);
      if(Image)
        delete Image;
      delete[] data;
    }
  } catch(handled_error const&) {
  } catch(std::bad_alloc&) {
    outOfMemory();
  }
  gl->remesh=true;

#ifndef HAVE_LIBOSMESA
#ifdef HAVE_LIBGLFW
  gl->redraw=true;
#endif

#ifdef HAVE_PTHREAD
  if(gl->thread && gl->readyAfterExport) {
    gl->readyAfterExport=false;
    gl->endwait(gl->readySignal,gl->readyLock);
  }
#endif
#endif
  exporting=false;
  gl->initSSBO=true;
}

void nodisplay()
{
}

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
  if(gl->thread) {
    gl->home();
#ifdef HAVE_PTHREAD
    if(!interact::interactive) {
      gl->endwait(gl->readySignal,gl->readyLock);
    }
#endif
    // Always signal the window to close in threaded mode
    glfwSetWindowShouldClose(gl->getGLFWWindow(), true);
    if(interact::interactive) {
      glfwHideWindow(gl->getGLFWWindow());
    }
  } else {
    ::glfwDestroyWindow(gl->getGLFWWindow());
    glfwTerminate();
    exit(0);
  }
#else
  // No windowing system available - just exit
  exit(0);
#endif
}

void AsyGLRender::cycleMode()
{
  // Call base class to handle mode cycling, ibl, and outlinemode
  AsyRender::cycleMode();

  // OpenGL-specific: restore nlights and set polygon mode
  remesh=true;
  if(ssbo)
    initSSBO=true;

  switch(mode) {
    case DRAWMODE_NORMAL: // regular
      nlights=nlights0;  // Restore original number of lights
      lastshader=-1;
      glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
      break;
    case DRAWMODE_OUTLINE: // outline
      nlights=0; // Force shader recompilation
      glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
      break;
    case DRAWMODE_WIREFRAME: // wireframe
      outlinemode=false;
      Nlights=1; // Force shader recompilation
      break;
  }
#ifndef HAVE_LIBOSMESA
  redraw=true;
#endif
}

#ifdef HAVE_LIBGLFW
bool capsize(int& width, int& height)
{
  bool resize=false;
  if(width > gl->screenWidth) {
    width=gl->screenWidth;
    resize=true;
  }
  if(height > gl->screenHeight) {
    height=gl->screenHeight;
    resize=true;
  }
  return resize;
}

void reshape0(int width, int height)
{
  gl_X=(gl_X/gl->Width)*width;
  gl_Y=(gl_Y/gl->Height)*height;

  gl->Width=width;
  gl->Height=height;

  static int lastWidth=1;
  static int lastHeight=1;
  if(gl->View && gl->Width*gl->Height > 1 && (gl->Width != lastWidth || gl->Height != lastHeight)
     && settings::verbose > 1) {
    cout << "Rendering " << stripDir(gl->Prefix) << " as "
         << gl->Width << "x" << gl->Height << " image" << endl;
    lastWidth=gl->Width;
    lastHeight=gl->Height;
  }

  glViewport(0,0,gl->Width,gl->Height);
  if(gl->ssbo)
    gl->initSSBO=true;
}

void windowposition(int& x, int& y, int width=-1, int height=-1)
{
  pair z=getSetting<pair>("position");
  x=(int) z.getx();
  y=(int) z.gety();
  if(x < 0) {
    x += gl->screenWidth-width;
    if(x < 0) x=0;
  }
  if(y < 0) {
    y += gl->screenHeight-height;
    if(y < 0) y=0;
  }
}

void setsize(int w, int h, bool reposition=true)
{
  int x,y;

  capsize(w,h);

  GLFWwindow* win = gl->getGLFWWindow();

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
  gl->redraw=true;
}

void capzoom()
{
  static double maxzoom=sqrt(DBL_MAX);
  static double minzoom=1.0/maxzoom;
  if(gl->Zoom <= minzoom) gl->Zoom=minzoom;
  if(gl->Zoom >= maxzoom) gl->Zoom=maxzoom;

  if(fabs(gl->Zoom-gl->lastzoom) > settings::getSetting<double>("zoomThreshold")) {
    gl->remesh=true;
    gl->lastzoom=gl->Zoom;
  }
}

void fullscreen(bool reposition=true)
{
  gl->Width=gl->screenWidth;
  gl->Height=gl->screenHeight;
  if(firstFit) {
    if(gl->Width < gl->Height*gl->Aspect)
      gl->Zoom *= gl->Width/(gl->Height*gl->Aspect);
    capzoom();
    firstFit=false;
  }
  gl->Xfactor=((double) gl->screenHeight)/gl->Height;
  gl->Yfactor=((double) gl->screenWidth)/gl->Width;
  reshape0(gl->Width,gl->Height);

  GLFWwindow* win = gl->getGLFWWindow();
  if(reposition)
    glfwSetWindowPos(win,0,0);
  glfwSetWindowSize(win,gl->Width,gl->Height);

  gl->redraw=true;
}

void fitscreen(bool reposition=true)
{
  switch(Fitscreen) {
    case 0: // Original size
    {
      gl->Xfactor=gl->Yfactor=1.0;
      double pixelRatio=getSetting<double>("devicepixelratio");
      setsize(oldWidth*pixelRatio,oldHeight*pixelRatio,reposition);
      break;
    }
    case 1: // Fit to screen in one dimension
    {
      int w=gl->screenWidth;
      int h=gl->screenHeight;
      if(w > h*gl->Aspect)
        w=std::min((int) ceil(h*gl->Aspect),w);
      else
        h=std::min((int) ceil(w/gl->Aspect),h);
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
  if(gl->thread && !interact::interactive)
    fitscreen(false);
}

stopWatch frameTimer;

void nextframe()
{
#ifdef HAVE_PTHREAD
  {
    gl->endwait(gl->readySignal,gl->readyLock);
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
  drawscene(gl->Width,gl->Height);
  if(fps) {
    if(gl->framecount < 20) // Measure steady-state framerate
      Timer.reset();
    else {
      double s=Timer.seconds(true);
      if(s > 0.0) {
        double rate=1.0/s;
        S.add(rate);
        if(gl->framecount % 20 == 0)
          cout << "FPS=" << rate << "\t" << S.mean() << " +/- " << S.stdev()
               << endl;
      }
    }
    ++gl->framecount;
  }

  glfwSwapBuffers(gl->getGLFWWindow());

  if(gl->queueExport) {
    Export();
    gl->queueExport=false;
  }
  if(!gl->thread) {
#if !defined(_WIN32)
    if(Oldpid != 0 && waitpid(Oldpid,NULL,WNOHANG) != Oldpid) {
      kill(Oldpid,SIGHUP);
      Oldpid=0;
    }
#endif
  }
}

void update()
{
  capzoom();

  gl->redraw=true;
#ifdef HAVE_RENDERER
  ::glfwShowWindow(gl->getGLFWWindow());
#endif

  // Call the AsyGLRender::update() method which handles view matrix computation
  gl->update();
}

void updateHandler(int)
{
  queueScreen=true;
  gl->remesh=true;
  update();

  glfwShowWindow(gl->getGLFWWindow());
}

// poll is no longer needed with GLFW - event handling is done in the main loop

void reshape(int width, int height)
{
  if(gl->thread) {
    static bool initialize=true;
    if(initialize) {
      initialize=false;
#if !defined(_WIN32)
      Signal(SIGUSR1,updateHandler);
#endif
    }
  }

  if(capsize(width,height)) {
    glfwSetWindowSize(gl->getGLFWWindow(),width,height);
  }

  reshape0(width,height);
  gl->remesh=true;
}

void exportHandler(int=0)
{
#ifndef HAVE_LIBOSMESA
#ifdef HAVE_LIBGLFW
  if(!Iconify) {
    glfwShowWindow(gl->getGLFWWindow());
  }
#endif
#endif
  gl->readyAfterExport=true;
  Export();

#ifndef HAVE_LIBOSMESA
#ifdef HAVE_LIBGLFW
  if(!Iconify)
    glfwHideWindow(gl->getGLFWWindow());
#endif
#endif
}

void init_osmesa()
{
#ifdef HAVE_LIBOSMESA
  // create context and buffer
  if(settings::verbose > 1)
    cout << "Allocating osmesa_buffer of size " << gl->screenWidth << "x"
         << gl->screenHeight << "x4x" << sizeof(GLubyte) << endl;
  osmesa_buffer=new unsigned char[gl->screenWidth*gl->screenHeight*4*sizeof(GLubyte)];
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
                        gl->screenWidth,gl->screenHeight )) {
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
  glUseProgram(gl->zeroShader);
  gl->lastshader=gl->zeroShader;
  glUniform1ui(glGetUniformLocation(gl->zeroShader,"width"),gl->Width);
  fpu_trap(false); // Work around FE_INVALID
  glDrawArrays(GL_TRIANGLES, 0, 3);
  fpu_trap(settings::trap());
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void compressCount()
{
  glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
  glUseProgram(gl->compressShader);
  gl->lastshader=gl->compressShader;
  glUniform1ui(glGetUniformLocation(gl->compressShader,"width"),gl->Width);
  fpu_trap(false); // Work around FE_INVALID
  glDrawArrays(GL_TRIANGLES, 0, 3);
  fpu_trap(settings::trap());
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void partialSums(bool readSize=false)
{
  // Compute partial sums on the GPU
  glUseProgram(gl->sum1Shader);
  glDispatchCompute(gl->g,1,1);

  if(gl->elements <= gl->groupSize*gl->groupSize)
    glUseProgram(gl->sum2fastShader);
  else {
    glUseProgram(gl->sum2Shader);
    glUniform1ui(glGetUniformLocation(gl->sum2Shader,"blockSize"),
                 ceilquotient(gl->g,gl->localSize));
  }
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
  glDispatchCompute(1,1,1);

  glUseProgram(gl->sum3Shader);
  glUniform1ui(glGetUniformLocation(gl->sum3Shader,"final"),gl->elements-1);
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
  glDispatchCompute(gl->g,1,1);
}

void resizeFragmentBuffer()
{
  if(GPUindexing) {
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER,gl->feedbackBuffer);
    GLuint *feedback=(GLuint *) glMapBuffer(GL_SHADER_STORAGE_BUFFER,GL_READ_ONLY);

    GLuint maxDepth=feedback[0];
    if(maxDepth > gl->maxSize)
      resizeBlendShader(maxDepth);

    gl->fragments=feedback[1];
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
  }

  if(gl->fragments > gl->maxFragments) {
    // Initialize the alpha buffer
    gl->maxFragments=11*gl->fragments/10;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER,gl->fragmentBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER,gl->maxFragments*sizeof(glm::vec4),
                 NULL,GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,4,gl->fragmentBuffer);


    glBindBuffer(GL_SHADER_STORAGE_BUFFER,gl->depthBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER,gl->maxFragments*sizeof(GLfloat),
                 NULL,GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,5,gl->depthBuffer);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER,gl->feedbackBuffer);
  }
}

void refreshBuffers()
{
  GLuint zero=0;
  gl->pixels=(gl->Width+1)*(gl->Height+1);

  if(gl->initSSBO) {
    gl->processors=1;

    GLuint Pixels;
    if(GPUindexing) {
      GLuint G=ceilquotient(gl->pixels,gl->groupSize);
      Pixels=gl->groupSize*G;

      GLuint globalSize=gl->localSize*ceilquotient(G,gl->localSize);
      glBindBuffer(GL_SHADER_STORAGE_BUFFER,gl->globalSumBuffer);
      glBufferData(GL_SHADER_STORAGE_BUFFER,globalSize*sizeof(GLuint),NULL,
                   GL_DYNAMIC_READ);
      glClearBufferData(GL_SHADER_STORAGE_BUFFER,GL_R32UI,GL_RED_INTEGER,
                        GL_UNSIGNED_INT,&zero);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER,3,gl->globalSumBuffer);
    } else Pixels=gl->pixels;

    glBindBuffer(GL_SHADER_STORAGE_BUFFER,gl->offsetBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER,(Pixels+2)*sizeof(GLuint),
                 NULL,GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,gl->offsetBuffer);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER,gl->countBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER,(Pixels+2)*sizeof(GLuint),
                 NULL,GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,2,gl->countBuffer);

    if(GPUcompress) {
      GLuint one=1;
      glBindBuffer(GL_ATOMIC_COUNTER_BUFFER,gl->elementsBuffer);
      glBufferData(GL_ATOMIC_COUNTER_BUFFER,sizeof(GLuint),&one,
                   GL_DYNAMIC_DRAW);
      glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER,0,gl->elementsBuffer);

      glBindBuffer(GL_SHADER_STORAGE_BUFFER,gl->indexBuffer);
      glBufferData(GL_SHADER_STORAGE_BUFFER,gl->pixels*sizeof(GLuint),
                   NULL,GL_DYNAMIC_DRAW);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER,1,gl->indexBuffer);
    }
    glClearBufferData(GL_SHADER_STORAGE_BUFFER,GL_R32UI,GL_RED_INTEGER,
                      GL_UNSIGNED_INT,&zero); // Clear count or index buffer

    glBindBuffer(GL_SHADER_STORAGE_BUFFER,gl->opaqueBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER,gl->pixels*sizeof(glm::vec4),NULL,
                 GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,6,gl->opaqueBuffer);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER,gl->opaqueDepthBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 sizeof(GLuint)+gl->pixels*sizeof(GLfloat),NULL,
                 GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,7,gl->opaqueDepthBuffer);
    const GLfloat zerof=0.0;
    glClearBufferData(GL_SHADER_STORAGE_BUFFER,GL_R32F,GL_RED,GL_FLOAT,&zerof);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER,gl->feedbackBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER,2*sizeof(GLuint),NULL,
                 GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,8,gl->feedbackBuffer);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER,gl->feedbackBuffer);
    gl->initSSBO=false;
  }

  // Determine the fragment offsets

  if(exporting && GPUindexing && !GPUcompress) {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER,gl->countBuffer);
    glClearBufferData(GL_SHADER_STORAGE_BUFFER,GL_R32UI,GL_RED_INTEGER,
                      GL_UNSIGNED_INT,&zero);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER,gl->feedbackBuffer);
  }

  if(!gl->interlock) {
    drawBuffer(material1Data,gl->countShader);
    drawBuffer(materialData,gl->countShader);
    drawBuffer(colorData,gl->countShader,true);
    drawBuffer(triangleData,gl->countShader,true);
  }

  glDepthMask(GL_FALSE); // Don't write to depth buffer
  glDisable(GL_MULTISAMPLE);
  drawBuffer(transparentData,gl->countShader,true);
  glEnable(GL_MULTISAMPLE);
  glDepthMask(GL_TRUE); // Write to depth buffer

  if(GPUcompress) {
    compressCount();
    GLuint *p=(GLuint *) glMapBuffer(GL_ATOMIC_COUNTER_BUFFER,GL_READ_WRITE);
    gl->elements=GPUindexing ? p[0] : p[0]-1;
    p[0]=1;
    glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER);
    if(gl->elements == 0) return;
  } else
    gl->elements=gl->pixels;

  if(GPUindexing) {
    gl->g=ceilquotient(gl->elements,gl->groupSize);
    gl->elements=gl->groupSize*gl->g;

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
      cout << "elements=" << gl->elements << endl;
      cout << "Tmin (ms)=" << T*1e3 << endl;
      cout << "Megapixels/second=" << gl->elements/T/1e6 << endl;
    }

    partialSums(true);
  } else {
    size_t size=gl->elements*sizeof(GLuint);

    // Compute partial sums on the CPU
    glBindBuffer(GL_SHADER_STORAGE_BUFFER,gl->countBuffer);
    GLuint *p=(GLuint *) glMapBufferRange(GL_SHADER_STORAGE_BUFFER,
                                          0,size+sizeof(GLuint),
                                              GL_MAP_READ_BIT);
    GLuint maxsize=p[0];
    GLuint *count=p+1;

    glBindBuffer(GL_SHADER_STORAGE_BUFFER,gl->offsetBuffer);
    GLuint *offset=(GLuint *) glMapBufferRange(GL_SHADER_STORAGE_BUFFER,
                                               sizeof(GLuint),size,
                                               GL_MAP_WRITE_BIT);

    size_t Offset=offset[0]=count[0];
    for(size_t i=1; i < gl->elements; ++i)
      offset[i]=Offset += count[i];
    gl->fragments=Offset;

    glBindBuffer(GL_SHADER_STORAGE_BUFFER,gl->offsetBuffer);
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER,gl->countBuffer);
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

    if(exporting) {
      glBindBuffer(GL_SHADER_STORAGE_BUFFER,gl->countBuffer);
      glClearBufferData(GL_SHADER_STORAGE_BUFFER,GL_R32UI,GL_RED_INTEGER,
                        GL_UNSIGNED_INT,&zero);
    } else
      clearCount();

    if(maxsize > gl->maxSize)
      resizeBlendShader(maxsize);
  }
  gl->lastshader=-1;
}

void setUniforms(vertexBuffer& data, GLint shader)
{
  bool normal=shader != gl->pixelShader;

  // Check if shader is valid
  if(shader == 0) {
    if(settings::verbose > 2) {
      cerr << "setUniforms: shader is 0!" << endl;
    }
    return;
  }

  if(shader != gl->lastshader) {
    glUseProgram(shader);

    if(normal)
      glUniform1ui(glGetUniformLocation(shader,"width"),gl->Width);
  }

  glUniformMatrix4fv(glGetUniformLocation(shader,"projViewMat"),1,GL_FALSE,
                     value_ptr(glm::mat4(gl->projViewMat)));

  glUniformMatrix4fv(glGetUniformLocation(shader,"viewMat"),1,GL_FALSE,
                     value_ptr(glm::mat4(gl->viewMat)));
  if(normal)
    glUniformMatrix3fv(glGetUniformLocation(shader,"normMat"),1,GL_FALSE,
                       value_ptr(normMat));

  if(shader == gl->countShader) {
    gl->lastshader=shader;
    return;
  }

  if(shader != gl->lastshader) {
    gl->lastshader=shader;
    glUniform1ui(glGetUniformLocation(shader,"nlights"),gl->nlights);

    for(size_t i=0; i < gl->nlights; ++i) {
      triple Lighti=gl->Lights[i];
      double *Diffusei=gl->Diffuse+4*i;
      glUniform3f(glGetUniformLocation(shader,
                                       getLightIndex(i,"direction").c_str()),
                  (GLfloat) Lighti.getx(),(GLfloat) Lighti.gety(),
                  (GLfloat) Lighti.getz());

      glUniform3f(glGetUniformLocation(shader,
                                       getLightIndex(i,"color").c_str()),
                  (GLfloat) Diffusei[0],(GLfloat) Diffusei[1],
                  (GLfloat) Diffusei[2]);
    }

    // IBL textures - disabled for now due to template issues
    // if(settings::getSetting<bool>("ibl")) {
    //   IBLbrdfTex.setUniform(glGetUniformLocation(shader, "reflBRDFSampler"));
    //   irradianceTex.setUniform(glGetUniformLocation(shader, "diffuseSampler"));
    //   reflTexturesTex.setUniform(glGetUniformLocation(shader, "reflImgSampler"));
    // }
  } else if (normal) {
    // Even if shader hasn't changed, update nlights uniform and light data if needed
    glUniform1ui(glGetUniformLocation(shader,"nlights"),gl->nlights);
    for(size_t i=0; i < gl->nlights; ++i) {
      triple Lighti=gl->Lights[i];
      double *Diffusei=gl->Diffuse+4*i;
      glUniform3f(glGetUniformLocation(shader,
                                       getLightIndex(i,"direction").c_str()),
                  (GLfloat) Lighti.getx(),(GLfloat) Lighti.gety(),
                  (GLfloat) Lighti.getz());

      glUniform3f(glGetUniformLocation(shader,
                                       getLightIndex(i,"color").c_str()),
                  (GLfloat) Diffusei[0],(GLfloat) Diffusei[1],
                  (GLfloat) Diffusei[2]);
    }
  }

  GLuint binding=0;
  GLint blockindex=glGetUniformBlockIndex(shader,"MaterialBuffer");
  glUniformBlockBinding(shader,blockindex,binding);
  bool copy=(gl->remesh || data.partial || !data.rendered) && !gl->copied;
  registerBuffer(data.materials,data.materialsBuffer,copy,GL_UNIFORM_BUFFER);
  glBindBufferBase(GL_UNIFORM_BUFFER,binding,data.materialsBuffer);
}

void drawBuffer(vertexBuffer& data, GLint shader, bool color)
{
  if(data.indices.empty()) return;

  // Ensure VAO is valid (non-zero) - should already be bound from setBuffers()
  if(gl->vao == 0) {
    if(settings::verbose > 2) {
      cerr << "drawBuffer: VAO not initialized! Creating now..." << endl;
    }
    glGenVertexArrays(1, &gl->vao);
    glBindVertexArray(gl->vao);  // Bind once and leave it bound
  }

  if(settings::verbose > 2) {
    cerr << "drawBuffer: gl->vao=" << gl->vao << endl;
  }

  // Check for OpenGL errors before drawing
  GLenum err = glGetError();
  if(err != GL_NO_ERROR && settings::verbose > 2) {
    cerr << "drawBuffer: OpenGL error at start: " << err << endl;
  }

  bool normal=shader != gl->pixelShader;

  const size_t size=sizeof(GLfloat);
  const size_t intsize=sizeof(GLint);
  const size_t bytestride=color ? sizeof(VertexData) :
    (normal ? sizeof(vertexData) : sizeof(vertexData0));

  // Debug output for material rendering
  if(settings::verbose > 2 && !data.vertices.empty()) {
    cerr << "drawBuffer: vertices.size=" << data.vertices.size()
         << " indices.size=" << data.indices.size()
         << " copy=" << ((gl->remesh || data.partial || !data.rendered) && !gl->copied) << endl;
  }

  // VAO is already bound from setBuffers(), no need to bind here

  bool copy=(gl->remesh || data.partial || !data.rendered) && !gl->copied;
  if(color) registerBuffer(data.Vertices,data.VerticesBuffer,copy);
  else if(normal) registerBuffer(data.vertices,data.verticesBuffer,copy);
  else registerBuffer(data.vertices0,data.vertices0Buffer,copy);

  registerBuffer(data.indices,data.indicesBuffer,copy,GL_ELEMENT_ARRAY_BUFFER);

  setUniforms(data,shader);

  data.rendered=true;

  glVertexAttribPointer(positionAttrib,3,GL_FLOAT,GL_FALSE,bytestride,
                        (void *) 0);
  glEnableVertexAttribArray(positionAttrib);

  if(normal && gl->Nlights > 0) {
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
  if(normal && gl->Nlights > 0)
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
  drawBuffer(material0Data,gl->pixelShader);
  material0Data.clear();
}

void drawMaterial1()
{
  drawBuffer(material1Data,gl->materialShader[Opaque]);
  material1Data.clear();
}

void drawMaterial()
{
  // Check for any pending OpenGL errors before drawing
  GLenum err = glGetError();
  if(err != GL_NO_ERROR && settings::verbose > 2) {
    cerr << "drawMaterial: OpenGL error before drawBuffer: " << err << endl;
  }

  drawBuffer(materialData,gl->materialShader[Opaque]);
  materialData.clear();
}

void drawColor()
{
  drawBuffer(colorData,gl->colorShader[Opaque],true);
  colorData.clear();
}

void drawTriangle()
{
  drawBuffer(triangleData,gl->generalShader[Opaque],true);
  triangleData.clear();
}

void aBufferTransparency()
{
  // Collect transparent fragments
  glDepthMask(GL_FALSE); // Disregard depth
  drawBuffer(transparentData,gl->transparentShader,true);
  glDepthMask(GL_TRUE); // Respect depth

  // Blend transparent fragments
  glDisable(GL_DEPTH_TEST);
  glUseProgram(gl->blendShader);
  gl->lastshader=gl->blendShader;
  glUniform1ui(glGetUniformLocation(gl->blendShader,"width"),gl->Width);
  glUniform4f(glGetUniformLocation(gl->blendShader,"background"),
              gl->Background[0],gl->Background[1],gl->Background[2],
              gl->Background[3]);
  fpu_trap(false); // Work around FE_INVALID
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
  glDrawArrays(GL_TRIANGLES,0,3);
  fpu_trap(settings::trap());
  transparentData.clear();
  glEnable(GL_DEPTH_TEST);
}

void drawTransparent()
{
  if(gl->ssbo) {
    glDisable(GL_MULTISAMPLE);
    aBufferTransparency();
    glEnable(GL_MULTISAMPLE);
  } else {
    sortTriangles();
    transparentData.rendered=false; // Force copying of sorted triangles to GPU
    glDepthMask(GL_FALSE); // Don't write to depth buffer
    drawBuffer(transparentData,gl->transparentShader,true);
    glDepthMask(GL_TRUE); // Write to depth buffer
    transparentData.clear();
  }
}

void drawBuffers()
{
  gl->copied=false;
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

  if(gl->ssbo) {
    if(transparent) {
      refreshBuffers();
      if(!gl->interlock) {
        resizeFragmentBuffer();
        gl->copied=true;
      }
    }
  }

  drawMaterial0();
  drawMaterial1();
  drawMaterial();
  drawColor();
  drawTriangle();

  if(transparent) {
    if(gl->ssbo)
      gl->copied=true;
    if(gl->interlock) resizeFragmentBuffer();
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

AsyGLRender::~AsyGLRender()
{
#ifdef HAVE_RENDERER
  if (this->View) {
    ::glfwDestroyWindow(getGLFWWindow());
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

  // Also set class member variables for use in setUniforms
  gl->Lights = args.lights;
  gl->Diffuse = args.diffuse;
  gl->Specular = args.specular;
  gl->Nlights = nlights;
  gl->nlights = nlights;
  gl->nlights0 = nlights;  // Save original for mode restoration

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

  haveScene = Xmin < Xmax && Ymin < Ymax && Zmin < Zmax;
  orthographic = Angle == 0.0;
  H = orthographic ? 0.0 : -tan(0.5 * Angle) * Zmax;
  Xfactor = Yfactor = 1.0;

  // Sync minimal globals with member variables (following Vulkan pattern)
  orthographic_gl = orthographic;

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

    // Make context current before GLEW initialization (matching reference pattern)
    glfwMakeContextCurrent(newWindow);

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
           << " glfwWindow=" << gl->glfwWindow << endl;
    }
  }
#endif
#endif

#if defined(HAVE_COMPUTE_SHADER) && !defined(HAVE_LIBOSMESA)
  GPUindexing=getSetting<bool>("GPUindexing");
  GPUcompress=getSetting<bool>("GPUcompress");
#else
  GPUindexing=false;
  GPUcompress=false;
#endif

  // Initialize OpenGL if needed
  if(initialize) {
    initialize = false;

#ifdef HAVE_LIBGLFW
#ifndef HAVE_LIBOSMESA
    // Verify the context is still current after window creation
    GLFWwindow* current = glfwGetCurrentContext();
    if(settings::verbose > 2) {
      cerr << "Post-window GLEW check: glfwWindow=" << gl->glfwWindow << " current=" << current << endl;
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
  if(!gl->ssbo) {
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  }

  mode = DRAWMODE_WIREFRAME;
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
    if(thread) {
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
        glfwGetCursorPos(getGLFWWindow(), &xpos, &ypos);
        xprev = xpos;
        yprev = ypos;
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
        if (yoffset > 0) Zoom *= zoomFactor;
        else Zoom /= zoomFactor;
    }
    capzoom();
    setProjection();
    update();
    redraw = true;
}

void AsyGLRender::onCursorPos(double xpos, double ypos)
{
    if (lastAction == "rotate") {
        Arcball arcball(xprev * 2 / Width - 1, 1 - yprev * 2 / Height,
                        xpos * 2 / Width - 1, 1 - ypos * 2 / Height);
        triple axis = arcball.axis;
        rotateMat = glm::rotate(2 * arcball.angle / Zoom * ArcballFactor,
                           glm::dvec3(axis.getx(), axis.gety(), axis.getz())) * rotateMat;
        update();
    } else if (lastAction == "shift") {
        shift(xpos - xprev, ypos - yprev);
        update();
    } else if (lastAction == "pan") {
        if (orthographic) shift(xpos - xprev, ypos - yprev);
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
    ::exitHandler(0);
}

void AsyGLRender::display()
{
#ifdef HAVE_RENDERER
  GLFWwindow* win = getGLFWWindow();
  if(View) {
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
  glfwSwapBuffers(getGLFWWindow());
#endif

  if(!thread) {
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
  // Update normal matrix for shaders (inverse transpose of view matrix rotation)
  dmat3 norm = dmat3(glm::inverse(this->viewMat));
  normMat = mat3(norm);  // Convert to float precision for shaders

  // Also update glBBT for billboard transformations
  const double *T=value_ptr(norm);
  for(size_t i=0; i < 9; ++i)
    glBBT[i]=T[i];
}

void AsyGLRender::update()
{
  capzoom();

  redraw=true;
#ifdef HAVE_RENDERER
  ::glfwShowWindow(getGLFWWindow());
#endif

  // Call base class update which has the correct view matrix computation (matching reference GLUT code)
  AsyRender::update();
}

void AsyGLRender::mainLoop()
{
#ifdef HAVE_RENDERER
  if(View) {
    GLFWwindow* win = getGLFWWindow();

    glfwRunLoop(win,
      [win](){ return !glfwWindowShouldClose(win); },
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
    if(thread) exportHandler();
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
