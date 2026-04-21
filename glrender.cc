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
#include <algorithm>

#if !defined(_WIN32)
#include <sys/time.h>
#include <unistd.h>
#endif

#include "common.h"
#include "locate.h"
#include "seconds.h"
#include "statistics.h"
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
double BBT[9] = {0};

Billboard BB;

// Accessor functions - directly access gl instance to avoid synchronization
const glm::dmat4& getProjViewMat()
{
  return gl->projViewMat;
}

const glm::dmat4& getViewMat()
{
  return gl->viewMat;
}

const glm::dmat3& getNormMat()
{
  return gl->normMat;
}

// Vertex buffers - these remain globals as they are populated by drawElement rendering
// Note: Using shared VertexBuffer type from render.h (library-agnostic)
// Vertex buffers are declared in render.h and defined in bezierpatch.cc / beziercurve.cc
// materialData, colorData, triangleData, transparentData - defined in bezierpatch.cc
// pointData, lineData - defined in beziercurve.cc

const size_t Nbuffer=10000;
const size_t nbuffer=1000;

size_t Maxmaterials;
size_t Nmaterials=1;
size_t nmaterials=48;

// Note: different name to avoid conflict with v3dheadertypes::orthographic enum
extern double Angle, Zoom0;
extern pair Shift, Margin;
extern double T[16], Tup[16];
extern double Xmin, Xmax, Ymin, Ymax, Zmin, Zmax;  // These are now member variables in AsyRender
static const double ASY_PI=acos(-1.0);
static const double ASY_DEGREES=180.0/ASY_PI;
static const double ASY_RADIANS=1.0/ASY_DEGREES;

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
string Action;

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

void noShaders()
{
  cerr << "GLSL shaders not found." << endl;
  exit(-1);
}

void AsyGLRender::initComputeShaders()
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
  s << "LOCALSIZE " << localSize << "u" << endl;
  shaderParams.push_back(s.str().c_str());
  s2 << "BLOCKSIZE " << blockSize << "u" << endl;
  shaderParams.push_back(s2.str().c_str());
  GLuint rc=compileAndLinkShader(shaders,shaderParams,true,false,true,false);
  if(rc == 0) {
    GPUindexing=false; // Compute shaders are unavailable.
    if(settings::verbose > 2)
      cout << "No compute shader support" << endl;
  } else {
//    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT,0,&maxgroups);
//    maxgroups=min(1024,maxgroups/(GLint) (localSize*blockSize));
    sum1Shader=rc;

    shaders[0]=ShaderfileModePair(sum2.c_str(),GL_COMPUTE_SHADER);
    sum2Shader=compileAndLinkShader(shaders,shaderParams,true,false,
                                          true);

    shaders[0]=ShaderfileModePair(sum2fast.c_str(),GL_COMPUTE_SHADER);
    sum2fastShader=compileAndLinkShader(shaders,shaderParams,true,false,
                                              true);

    shaders[0]=ShaderfileModePair(sum3.c_str(),GL_COMPUTE_SHADER);
    sum3Shader=compileAndLinkShader(shaders,shaderParams,true,false,
                                          true);
  }
}

void AsyGLRender::initBlendShader()
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
  blendShader=compileAndLinkShader(shaders,shaderParams,ssbo);
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

void AsyGLRender::setBuffers()
{
  if(settings::verbose > 2) {
    cerr << "setBuffers: Creating VAO, vao=" << vao << endl;
  }
  glGenVertexArrays(1,&vao);
  if(settings::verbose > 2) {
    cerr << "setBuffers: VAO created, vao=" << vao << endl;
  }
  // Bind VAO once and leave it bound for all subsequent draw operations
  glBindVertexArray(vao);

  // Buffers are pre-sized as needed, no explicit reserve calls needed
  materialData.renderCount=0;
  colorData.renderCount=0;
  triangleData.renderCount=0;
  transparentData.renderCount=0;

  // Create materials uniform buffer
  glGenBuffers(1, &materialsBuffer);

#ifdef HAVE_SSBO
  glGenBuffers(1, &offsetBuffer);
  if(GPUindexing)
    glGenBuffers(1, &globalSumBuffer);
  glGenBuffers(1, &feedbackBuffer);
  glGenBuffers(1, &countBuffer);
  if(GPUcompress) {
    glGenBuffers(1, &indexBuffer);
    glGenBuffers(1, &elementsBuffer);
  }
  glGenBuffers(1, &fragmentBuffer);
  glGenBuffers(1, &depthBuffer);
  glGenBuffers(1, &opaqueBuffer);
  glGenBuffers(1, &opaqueDepthBuffer);
#endif

  if(settings::verbose > 2) {
    cerr << "setBuffers: Done, vao=" << vao << endl;
  }
}

void AsyGLRender::initShaders()
{
  Nlights = nlights == 0 ? 0 : std::max(Nlights, nlights);
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

  if(ibl) {
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
  countShader=compileAndLinkShader(shaders,shaderParams,
                                         true,false,false,true);
  if(countShader)
    shaderParams.push_back("HAVE_SSBO");
#else
  countShader=0;
#endif

  ssbo=countShader;
  if(!ssbo) {
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  }

#ifdef HAVE_LIBOSMESA
  interlock=false;
#else
  interlock=ssbo && getSetting<bool>("GPUinterlock");
#endif

  if(!ssbo && settings::verbose > 2)
    cout << "No SSBO support; order-independent transparency unavailable"
         << endl;

  shaders[1]=ShaderfileModePair(fragment.c_str(),GL_FRAGMENT_SHADER);
  shaderParams.push_back("MATERIAL");
  if(orthographic)
    shaderParams.push_back("ORTHOGRAPHIC");

  ostringstream lights,materials,opaque;
  lights << "Nlights " << Nlights;
  shaderParams.push_back(lights.str().c_str());
  materials << "Nmaterials " << Nmaterials;
  shaderParams.push_back(materials.str().c_str());

  shaderParams.push_back("WIDTH");
  pixelShader=compileAndLinkShader(shaders,shaderParams,ssbo);
  shaderParams.pop_back();

  shaderParams.push_back("NORMAL");
  if(interlock) shaderParams.push_back("HAVE_INTERLOCK");
  materialShader[0]=compileAndLinkShader(shaders,shaderParams,
                                               ssbo,interlock,false,true);
  if(interlock && !materialShader[0]) {
    shaderParams.pop_back();
    interlock=false;
    materialShader[0]=compileAndLinkShader(shaders,shaderParams,ssbo);
    if(settings::verbose > 2)
      cout << "No fragment shader interlock support" << endl;
  }

  shaderParams.push_back("OPAQUE");
  materialShader[1]=compileAndLinkShader(shaders,shaderParams,ssbo);
  shaderParams.pop_back();

  shaderParams.push_back("COLOR");
  colorShader[0]=compileAndLinkShader(shaders,shaderParams,ssbo,
                                            interlock);
  shaderParams.push_back("OPAQUE");
  colorShader[1]=compileAndLinkShader(shaders,shaderParams,ssbo);
  shaderParams.pop_back();

  shaderParams.push_back("GENERAL");
  if(mode != DRAWMODE_NORMAL)
    shaderParams.push_back("WIREFRAME");
  generalShader[0]=compileAndLinkShader(shaders,shaderParams,ssbo,
                                              interlock);
  shaderParams.push_back("OPAQUE");
  generalShader[1]=compileAndLinkShader(shaders,shaderParams,ssbo);
  shaderParams.pop_back();

  shaderParams.push_back("TRANSPARENT");
  transparentShader=compileAndLinkShader(shaders,shaderParams,ssbo,
                                               interlock);
  shaderParams.clear();

  if(ssbo) {
    if(GPUindexing)
      shaderParams.push_back("GPUINDEXING");
    shaders[0]=ShaderfileModePair(screen.c_str(),GL_VERTEX_SHADER);
    shaders[1]=ShaderfileModePair(compress.c_str(),GL_FRAGMENT_SHADER);
    compressShader=compileAndLinkShader(shaders,shaderParams,ssbo);
    if(GPUindexing)
      shaderParams.pop_back();
    else {
      shaders[1]=ShaderfileModePair(zero.c_str(),GL_FRAGMENT_SHADER);
      zeroShader=compileAndLinkShader(shaders,shaderParams,ssbo);
    }
    maxSize=1;
    initBlendShader();
  }
  lastshader=-1;

  if(vao == 0)
    setBuffers();
}

void AsyGLRender::deleteComputeShaders()
{
  glDeleteProgram(sum1Shader);
  glDeleteProgram(sum2Shader);
  glDeleteProgram(sum2fastShader);
  glDeleteProgram(sum3Shader);
}

void AsyGLRender::deleteBlendShader()
{
  glDeleteProgram(blendShader);
}

void AsyGLRender::deleteShaders()
{
  if(ssbo) {
    deleteBlendShader();
    if(GPUindexing)
      deleteComputeShaders();
    else
      glDeleteProgram(zeroShader);
    glDeleteProgram(countShader);
    glDeleteProgram(compressShader);
  }

  if (transparentShader != 0)
    glDeleteProgram(transparentShader);
  for(unsigned int opaque=0; opaque < 2; ++opaque) {
    if (generalShader[opaque] != 0)
      glDeleteProgram(generalShader[opaque]);
    if (colorShader[opaque] != 0)
      glDeleteProgram(colorShader[opaque]);
    if (materialShader[opaque] != 0)
      glDeleteProgram(materialShader[opaque]);
  }
  if (pixelShader != 0)
    glDeleteProgram(pixelShader);
}

void AsyGLRender::resizeBlendShader(GLuint maxsize)
{
  maxSize=ceilpow2(maxsize);
  deleteBlendShader();
  initBlendShader();
}

void AsyGLRender::drawFrame()
{
  if((nlights == 0 && Nlights > 0) || nlights > Nlights ||
     nmaterials > Nmaterials) {
    deleteShaders();
    initShaders();
  }

  // Set viewport before clearing (in case it wasn't set)
  // Skip during export - trBeginTile handles viewport for tiling
  if(!exporting)
    glViewport(0, 0, Width, Height);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // Use member variables from AsyGLRender (following Vulkan pattern)
  if(xmin >= xmax || ymin >= ymax || Zmin >= Zmax) return;

#ifdef HAVE_RENDERER
  drawBuffers();
#endif
}

// Return x divided by y rounded up to the nearest integer.
int ceilquotient(int x, int y)
{
  return (x+y-1)/y;
}

void AsyGLRender::Export(int)
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

      // Use member variables from AsyGLRender (following Vulkan pattern)
      double dXmin = xmin;
      double dXmax = xmax;
      double dYmin = ymin;
      double dYmax = ymax;
      double dZmin = Zmin;
      double dZmax = Zmax;

      size_t count=0;
      if(haveScene) {
        (orthographic ? trOrtho : trFrustum)(tr,dXmin,dXmax,dYmin,dYmax,-dZmax,-dZmin);
        do {
          trBeginTile(tr);
          remesh=true;
          prepareScene();
          drawFrame();
          lastshader=-1;
          ++count;
        } while (trEndTile(tr));
      } else {// clear screen and return
        prepareScene();
        drawFrame();
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

#ifndef HAVE_LIBOSMESA
#ifdef HAVE_LIBGLFW
  redraw=true;
#endif

#ifdef HAVE_PTHREAD
  if(thread && readyAfterExport) {
    readyAfterExport=false;
    endwait(readySignal,readyLock);
  }
#endif
#endif
  exporting=false;
  initSSBO=true;
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
#else
  gl->quit();
#endif
}

void AsyGLRender::cycleMode()
{
  // Call base class to handle mode cycling and ibl
  AsyRender::cycleMode();

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
      Nlights=1; // Force shader recompilation
      break;
  }
}

#ifdef HAVE_LIBGLFW

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

void AsyGLRender::clearCount()
{
  glUseProgram(zeroShader);
  lastshader=zeroShader;
  glUniform1ui(glGetUniformLocation(zeroShader,"width"),Width);
  fpu_trap(false); // Work around FE_INVALID
  glDrawArrays(GL_TRIANGLES, 0, 3);
  fpu_trap(settings::trap());
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void AsyGLRender::compressCount()
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

void AsyGLRender::partialSums(bool readSize)
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

void AsyGLRender::resizeFragmentBuffer()
{
  if(GPUindexing) {
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER,feedbackBuffer);
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
    glBindBuffer(GL_SHADER_STORAGE_BUFFER,fragmentBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER,maxFragments*sizeof(glm::vec4),
                 NULL,GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,4,fragmentBuffer);


    glBindBuffer(GL_SHADER_STORAGE_BUFFER,depthBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER,maxFragments*sizeof(GLfloat),
                 NULL,GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,5,depthBuffer);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER,feedbackBuffer);
  }
}

void AsyGLRender::refreshBuffers()
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
      glBindBuffer(GL_SHADER_STORAGE_BUFFER,globalSumBuffer);
      glBufferData(GL_SHADER_STORAGE_BUFFER,globalSize*sizeof(GLuint),NULL,
                   GL_DYNAMIC_READ);
      glClearBufferData(GL_SHADER_STORAGE_BUFFER,GL_R32UI,GL_RED_INTEGER,
                        GL_UNSIGNED_INT,&zero);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER,3,globalSumBuffer);
    } else Pixels=pixels;

    glBindBuffer(GL_SHADER_STORAGE_BUFFER,offsetBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER,(Pixels+2)*sizeof(GLuint),
                 NULL,GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,offsetBuffer);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER,countBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER,(Pixels+2)*sizeof(GLuint),
                 NULL,GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,2,countBuffer);

    if(GPUcompress) {
      GLuint one=1;
      glBindBuffer(GL_ATOMIC_COUNTER_BUFFER,elementsBuffer);
      glBufferData(GL_ATOMIC_COUNTER_BUFFER,sizeof(GLuint),&one,
                   GL_DYNAMIC_DRAW);
      glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER,0,elementsBuffer);

      glBindBuffer(GL_SHADER_STORAGE_BUFFER,indexBuffer);
      glBufferData(GL_SHADER_STORAGE_BUFFER,pixels*sizeof(GLuint),
                   NULL,GL_DYNAMIC_DRAW);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER,1,indexBuffer);
    }
    glClearBufferData(GL_SHADER_STORAGE_BUFFER,GL_R32UI,GL_RED_INTEGER,
                      GL_UNSIGNED_INT,&zero); // Clear count or index buffer

    glBindBuffer(GL_SHADER_STORAGE_BUFFER,opaqueBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER,pixels*sizeof(glm::vec4),NULL,
                 GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,6,opaqueBuffer);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER,opaqueDepthBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 sizeof(GLuint)+pixels*sizeof(GLfloat),NULL,
                 GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,7,opaqueDepthBuffer);
    const GLfloat zerof=0.0;
    glClearBufferData(GL_SHADER_STORAGE_BUFFER,GL_R32F,GL_RED,GL_FLOAT,&zerof);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER,feedbackBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER,2*sizeof(GLuint),NULL,
                 GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,8,feedbackBuffer);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER,feedbackBuffer);
    initSSBO=false;
  }

  // Determine the fragment offsets

  if(exporting && GPUindexing && !GPUcompress) {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER,countBuffer);
    glClearBufferData(GL_SHADER_STORAGE_BUFFER,GL_R32UI,GL_RED_INTEGER,
                      GL_UNSIGNED_INT,&zero);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER,feedbackBuffer);
  }

  if(!interlock) {
    drawBuffer(lineData,countShader,false,4);
    drawBuffer(materialData,countShader,false,4);
    drawBuffer(colorData,countShader,true,4);
    drawBuffer(triangleData,countShader,true,4);
  }

  glDepthMask(GL_FALSE); // Don't write to depth buffer
  glDisable(GL_MULTISAMPLE);
  drawBuffer(transparentData,countShader,true,4);
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
    glBindBuffer(GL_SHADER_STORAGE_BUFFER,countBuffer);
    GLuint *p=(GLuint *) glMapBufferRange(GL_SHADER_STORAGE_BUFFER,
                                          0,size+sizeof(GLuint),
                                              GL_MAP_READ_BIT);
    GLuint maxsize=p[0];
    GLuint *count=p+1;

    glBindBuffer(GL_SHADER_STORAGE_BUFFER,offsetBuffer);
    GLuint *offset=(GLuint *) glMapBufferRange(GL_SHADER_STORAGE_BUFFER,
                                               sizeof(GLuint),size,
                                               GL_MAP_WRITE_BIT);

    size_t Offset=offset[0]=count[0];
    for(size_t i=1; i < elements; ++i)
      offset[i]=Offset += count[i];
    fragments=Offset;

    glBindBuffer(GL_SHADER_STORAGE_BUFFER,offsetBuffer);
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER,countBuffer);
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

    if(exporting) {
      glBindBuffer(GL_SHADER_STORAGE_BUFFER,countBuffer);
      glClearBufferData(GL_SHADER_STORAGE_BUFFER,GL_R32UI,GL_RED_INTEGER,
                        GL_UNSIGNED_INT,&zero);
    } else
      clearCount();

    if(maxsize > maxSize)
      resizeBlendShader(maxsize);
  }
  lastshader=-1;
}

void AsyGLRender::setUniformsOpenGL(GLint shader)
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
      double *Diffusei= LightsDiffuse+4*i;
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
  }

  // Bind global materials buffer
  GLuint binding=0;
  GLuint blockindex=glGetUniformBlockIndex(shader,"MaterialBuffer");
  if(blockindex != GL_INVALID_INDEX) {
    glUniformBlockBinding(shader,blockindex,binding);
    bool copy=shouldUpdateBuffers;
    registerBuffer(materials, materialsBuffer, copy, GL_UNIFORM_BUFFER);
    shouldUpdateBuffers=false;
    glBindBufferBase(GL_UNIFORM_BUFFER, binding, materialsBuffer);
  }
}

void AsyGLRender::drawBuffer(VertexBuffer& data, GLint shader, bool color, unsigned int drawType)  // drawType: 0=GL_POINTS, 1=GL_LINES, 4=GL_TRIANGLES
{
  if(data.indices.empty()) return;

  // Check for OpenGL errors before drawing
  GLenum err = glGetError();
  if(err != GL_NO_ERROR && settings::verbose > 2) {
    cerr << "drawBuffer: OpenGL error at start: " << err << endl;
  }

  bool normal=shader != pixelShader;

  // Determine which vertex vector to use and the stride
  size_t bytestride = 0;
  size_t nvertices = 0;

  if(color) {
    bytestride = sizeof(ColorVertex);
    nvertices = data.colorVertices.size();
  } else if(normal) {
    bytestride = sizeof(MaterialVertex);
    nvertices = data.materialVertices.size();
  } else {
    bytestride = sizeof(PointVertex);
    nvertices = data.pointVertices.size();
  }

  // Debug output for material rendering
  if(settings::verbose > 2 && nvertices > 0) {
    cerr << "drawBuffer: vertices.size=" << nvertices
         << " indices.size=" << data.indices.size() << endl;
  }

  // VAO is already bound from setBuffers(), no need to bind here

  GLuint vertexBuffer = 0;
  glGenBuffers(1, &vertexBuffer);
  glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);

  if(color) {
    glBufferData(GL_ARRAY_BUFFER, data.colorVertices.size() * sizeof(ColorVertex),
                 data.colorVertices.data(), GL_DYNAMIC_DRAW);
  } else if(normal) {
    glBufferData(GL_ARRAY_BUFFER, data.materialVertices.size() * sizeof(MaterialVertex),
                 data.materialVertices.data(), GL_DYNAMIC_DRAW);
  } else {
    glBufferData(GL_ARRAY_BUFFER, data.pointVertices.size() * sizeof(PointVertex),
                 data.pointVertices.data(), GL_DYNAMIC_DRAW);
  }

  GLuint indexBuffer = 0;
  glGenBuffers(1, &indexBuffer);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, data.indices.size() * sizeof(uint32_t),
               data.indices.data(), GL_DYNAMIC_DRAW);

  setUniformsOpenGL(shader);

  data.renderCount++;

  // Position attribute (3 floats)
  if(color) {
    glVertexAttribPointer(positionAttrib, 3, GL_FLOAT, GL_FALSE, bytestride,
                          (void *) offsetof(ColorVertex, position));
  } else if(normal) {
    glVertexAttribPointer(positionAttrib, 3, GL_FLOAT, GL_FALSE, bytestride,
                          (void *) offsetof(MaterialVertex, position));
  } else {
    glVertexAttribPointer(positionAttrib, 3, GL_FLOAT, GL_FALSE, bytestride,
                          (void *) offsetof(PointVertex, position));
  }
  glEnableVertexAttribArray(positionAttrib);

  if(normal && nlights > 0) {
    // Normal attribute (3 floats)
    glVertexAttribPointer(normalAttrib, 3, GL_FLOAT, GL_FALSE, bytestride,
                          (void *) offsetof(MaterialVertex, normal));
    glEnableVertexAttribArray(normalAttrib);
  } else if(!normal) {
    // Width attribute for points (1 float)
    glVertexAttribPointer(widthAttrib, 1, GL_FLOAT, GL_FALSE, bytestride,
                          (void *) offsetof(PointVertex, width));
    glEnableVertexAttribArray(widthAttrib);
  }

  // Material index attribute (1 int)
  if(color) {
    glVertexAttribIPointer(materialAttrib, 1, GL_INT, bytestride,
                           (void *) offsetof(ColorVertex, material));
  } else if(normal) {
    glVertexAttribIPointer(materialAttrib, 1, GL_INT, bytestride,
                           (void *) offsetof(MaterialVertex, material));
  } else {
    glVertexAttribIPointer(materialAttrib, 1, GL_INT, bytestride,
                           (void *) offsetof(PointVertex, material));
  }
  glEnableVertexAttribArray(materialAttrib);

  if(color) {
    // Color attribute (4 floats)
    glVertexAttribPointer(colorAttrib, 4, GL_FLOAT, GL_FALSE, bytestride,
                          (void *) offsetof(ColorVertex, color));
    glEnableVertexAttribArray(colorAttrib);
  }

  fpu_trap(false); // Work around FE_INVALID
  glDrawElements(drawType, data.indices.size(), GL_UNSIGNED_INT, (void *) 0);

  // Check for OpenGL errors after draw call
  GLenum err2 = glGetError();
  if(err2 != GL_NO_ERROR && settings::verbose > 1) {
    cerr << "drawBuffer: OpenGL error after glDrawElements: " << err2 << endl;
  }

  fpu_trap(settings::trap());

  // Disable attribute arrays but keep VAO bound for next draw call
  glDisableVertexAttribArray(positionAttrib);
  if(normal && nlights > 0)
    glDisableVertexAttribArray(normalAttrib);
  if(!normal)
    glDisableVertexAttribArray(widthAttrib);
  glDisableVertexAttribArray(materialAttrib);
  if(color)
    glDisableVertexAttribArray(colorAttrib);

  glDeleteBuffers(1, &vertexBuffer);
  glDeleteBuffers(1, &indexBuffer);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void AsyGLRender::drawMaterial0()
{
  drawBuffer(pointData,pixelShader,false,0);  // GL_POINTS
  pointData.clear();
}

void AsyGLRender::drawMaterial1()
{
  drawBuffer(lineData,materialShader[Opaque],false,1);  // GL_LINES
  lineData.clear();
}

void AsyGLRender::drawMaterial()
{
  drawBuffer(materialData,materialShader[Opaque]);  // default GL_TRIANGLES
  materialData.clear();
}

void AsyGLRender::drawColor()
{
  drawBuffer(colorData,colorShader[Opaque],true);  // default GL_TRIANGLES
  colorData.clear();
}

void AsyGLRender::drawTriangle()
{
  drawBuffer(triangleData,generalShader[Opaque],true);  // default GL_TRIANGLES
  triangleData.clear();
}

void AsyGLRender::aBufferTransparency()
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
              Background[0],Background[1],Background[2],
              Background[3]);
  fpu_trap(false); // Work around FE_INVALID
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
  glDrawArrays(GL_TRIANGLES,0,3);
  fpu_trap(settings::trap());
  // Don't clear transparentData - it needs to persist for subsequent frames
  glEnable(GL_DEPTH_TEST);
}

void AsyGLRender::drawTransparent()
{
  if(ssbo) {
    glDisable(GL_MULTISAMPLE);
    aBufferTransparency();
    glEnable(GL_MULTISAMPLE);
  } else {
    // Sort transparent triangles by depth (simplified - just draw directly)
    glDepthMask(GL_FALSE); // Don't write to depth buffer
    drawBuffer(transparentData,transparentShader,true,4);
    glDepthMask(GL_TRUE); // Write to depth buffer
    // Don't clear transparentData - it needs to persist for subsequent frames
  }
}

void AsyGLRender::drawBuffers()
{
  Opaque=transparentData.indices.empty();
  bool transparent=!Opaque;

  if(settings::verbose > 2) {
    cerr << "drawBuffers: Opaque=" << Opaque
         << " point indices=" << pointData.indices.size()
         << " line indices=" << lineData.indices.size()
         << " material indices=" << materialData.indices.size()
         << " color indices=" << colorData.indices.size()
         << " triangle indices=" << triangleData.indices.size()
         << " transparent indices=" << transparentData.indices.size()
         << endl;
  }

  if(ssbo) {
    if(transparent) {
      refreshBuffers();
      // Reset copiedThisFrame after count pass so render pass can upload
      pointData.copiedThisFrame = false;
      lineData.copiedThisFrame = false;
      materialData.copiedThisFrame = false;
      colorData.copiedThisFrame = false;
      triangleData.copiedThisFrame = false;
      transparentData.copiedThisFrame = false;

      if(!interlock) {
        resizeFragmentBuffer();
      }
    }
  }

  drawMaterial0();
  drawMaterial1();
  drawMaterial();
  drawColor();
  drawTriangle();

  if(transparent) {
    if(interlock) resizeFragmentBuffer();
    drawTransparent();
  }
  Opaque=0;
}

AsyGLRender::~AsyGLRender()
{
#ifdef HAVE_RENDERER
  if (this->View) {
    ::glfwDestroyWindow(getRenderWindow());
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
  Iconify=getSetting<bool>("iconify");

#if !defined(_WIN32)
  setenv("XMODIFIERS","",true);
#endif

  bool v3d=args.format == "v3d";
  bool webgl=args.format == "html";
  bool format3d=webgl || v3d;

  Prefix = args.prefix;
  pic = args.pic;
  Format = args.format;
  nlights = args.nlightsin;

  Lights = args.lights;
  LightsDiffuse = args.diffuse;
  Specular = args.specular;

  nlights0 = nlights;  // Save original for mode restoration

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

  haveScene = Xmin < Xmax && Ymin < Ymax && Zmin < Zmax;
  orthographic = Angle == 0.0;
  H = orthographic ? 0.0 : -tan(0.5 * Angle) * Zmax;

  Xfactor = Yfactor = 1.0;

  pair maxtile=getSetting<pair>("maxtile");
  maxTileWidth=(int) maxtile.getx();
  maxTileHeight=(int) maxtile.gety();
  if(maxTileWidth <= 0) maxTileWidth=1024;
  if(maxTileHeight <= 0) maxTileHeight=768;

#ifdef HAVE_PTHREAD
  static bool initializedView=false;

#ifdef HAVE_LIBOSMESA
  if(!webgl) {
    screenWidth=maxTileWidth;
    screenHeight=maxTileHeight;

    static bool osmesa_initialized=false;
    if(!osmesa_initialized) {
      osmesa_initialized=true;
      fpu_trap(false); // Work around FE_INVALID.
      init_osmesa();
      fpu_trap(settings::trap());
    }
  }
#else
  if(!initialized)
    Fitscreen=1;
#endif
#endif

  for(int i=0; i < 16; ++i)
    T[i]=args.t[i];

  for(int i=0; i < 16; ++i)
    Tup[i]=args.tup[i];

  if(!(initialized && interact::interactive)) {
    antialias=settings::getSetting<Int>("antialias") > 1;
    double expand;
    if(format3d)
      expand=1.0;
    else {
      expand=settings::getSetting<double>("render");
      if(expand < 0)
        expand *= (Format.empty() || Format == "eps" || Format == "pdf")                 ? -2.0 : -1.0;
      if(antialias) expand *= 2.0;
    }

    oWidth=args.width;
    oHeight=args.height;
    Aspect=args.width/args.height;

    // Force a hard viewport limit to work around direct rendering bugs.
    // Alternatively, one can use -glOptions=-indirect (with a performance
    // penalty).
    pair maxViewport = settings::getSetting<pair>("maxviewport");
    int maxWidth = maxViewport.getx() > 0 ? (int)ceil(maxViewport.getx()) : screenWidth;
    int maxHeight = maxViewport.gety() > 0 ? (int)ceil(maxViewport.gety()) : screenHeight;
    if(maxWidth <= 0) maxWidth = max(maxHeight, 2);
    if(maxHeight <= 0) maxHeight = max(maxWidth, 2);

    if(screenWidth <= 0) screenWidth=maxWidth;
    else screenWidth=min(screenWidth,maxWidth);
    if(screenHeight <= 0) screenHeight=maxHeight;
    else screenHeight=min(screenHeight,maxHeight);

    fullWidth=(int) ceil(expand*args.width);
    fullHeight=(int) ceil(expand*args.height);

    if(format3d) {
      Width=fullWidth;
      Height=fullHeight;
    } else {
#ifdef HAVE_RENDERER
      GLFWmonitor* monitor=NULL;
      glfwInit();
      monitor=glfwGetPrimaryMonitor();
      if(monitor) {
        int mx, my;
        glfwGetMonitorWorkarea(monitor, &mx, &my, &screenWidth, &screenHeight);
      } else
#endif
        {
          screenWidth=fullWidth;
          screenHeight=fullHeight;
        }

      Width=min(fullWidth,screenWidth);
      Height=min(fullHeight,screenHeight);

      if(Width > Height*Aspect)
        Width=min((int) (ceil(Height*Aspect)),screenWidth);
      else
        Height=min((int) (ceil(Width/Aspect)),screenHeight);
    }


#ifdef HAVE_RENDERER
    home(format3d);
#endif
    if(format3d) {
      remesh=true;
      return;
    }
    maxFragments=0;

    ArcballFactor=1+8.0*hypot(Margin.getx(),Margin.gety())/hypot(Width,Height);
    Aspect=((double) Width)/Height;

#ifdef HAVE_RENDERER
    setosize();
#endif
  }

#ifdef HAVE_RENDERER
  havewindow=initialized && thread;

  if(thread && format3d)
    format3dWait=true;

  clearMaterials();
  shouldUpdateBuffers=true;
  initialized=true;
#endif

#ifdef HAVE_PTHREAD
  if(thread && initializedView) {
    if(View) {
      // Called from asymain thread, main thread handles rendering
      hideWindow=false;
      messageQueue.enqueue(RendererMessage::updateRenderer);
    } else readyAfterExport=queueExport=true;
    return;
  }
#endif

  // Create GLFW window BEFORE OpenGL initialization if viewing and not using OSMesa
#ifdef HAVE_LIBGLFW
#ifndef HAVE_LIBOSMESA
  if(View && glfwWindow == nullptr) {
    // Use appropriate window size - for hidden windows use maxTile dimensions
    int winWidth = Width;
    int winHeight = Height;
//    if(!View || Iconify) {
//      // For hidden/offscreen rendering, use larger tile dimensions
//      winWidth = maxTileWidth > 0 ? maxTileWidth : 1024;
//      winHeight = maxTileHeight > 0 ? maxTileHeight : 768;
//    }

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
           << " glfwWindow=" << glfwWindow << endl;
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

  // Initialize GPU compute parameters (must be done before any call to initShaders())
  if(GPUindexing) {
    localSize = settings::getSetting<Int>("GPUlocalSize");
    checkpow2(localSize,"GPUlocalSize");
    blockSize = settings::getSetting<Int>("GPUblockSize");
    checkpow2(blockSize,"GPUblockSize");
    groupSize = localSize * blockSize;
  }

  // Initialize OpenGL if needed
  if(!initialized) {
    initialized = true;

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
    initializedView = true;
  }
#endif

  glEnable(GL_DEPTH_TEST);

  mode = DRAWMODE_WIREFRAME;
  cycleMode();

  ViewExport = View;
#ifdef HAVE_LIBOSMESA
  View = false;
#endif

#ifdef HAVE_RENDERER
  havewindow = initialized && thread;
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
        glfwGetCursorPos(getRenderWindow(), &xpos, &ypos);
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
    reshape(width, height);
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

/**
 * Show the window if hidden (GLFW-specific implementation).
 */
void AsyGLRender::showWindow()
{
  GLFWwindow* win = getRenderWindow();

  if(View && !hideWindow && !glfwGetWindowAttrib(win,GLFW_VISIBLE))
    ::glfwShowWindow(win);
}

/**
 * Swap front and back buffers (GLFW-specific implementation).
 */
void AsyGLRender::swapBuffers()
{
#ifdef HAVE_RENDERER
  glfwSwapBuffers(getRenderWindow());
#endif
}

void frustum(GLdouble left, GLdouble right, GLdouble bottom,
             GLdouble top, GLdouble nearVal, GLdouble farVal)
{
  gl->frustum(left, right, bottom, top, nearVal, farVal);
}

void ortho(GLdouble left, GLdouble right, GLdouble bottom,
           GLdouble top, GLdouble nearVal, GLdouble farVal)
{
  gl->ortho(left, right, bottom, top, nearVal, farVal);
}

void AsyGLRender::updateModelViewData()
{
  AsyRender::updateModelViewData();

  // Update BBT array for Billboard transformations (using normMat directly)
  const double *T=value_ptr(this->normMat);
  for(size_t i=0; i < 9; ++i)
    BBT[i]=T[i];
}

void AsyGLRender::update()
{
  capzoom();

  redraw=true;
#ifdef HAVE_RENDERER
  if(glfwWindow)
    ::glfwShowWindow(getRenderWindow());
#endif

  // Call base class update which has the correct view matrix computation (matching reference GLUT code)
  AsyRender::update();
}

#ifdef HAVE_RENDERER
GLFWwindow* AsyGLRender::getRenderWindow() const
{
  return static_cast<GLFWwindow*>(glfwWindow);
}
#endif

void AsyGLRender::exportHandler(int)
{
#ifdef HAVE_RENDERER
  readyAfterExport=true;
#ifndef HAVE_LIBOSMESA
#ifdef HAVE_LIBGLFW
  if(glfwWindow && !Iconify) {
    glfwShowWindow(getRenderWindow());
  }
#endif
#endif
  Export();

#ifndef HAVE_LIBOSMESA
#ifdef HAVE_LIBGLFW
  if(glfwWindow && !Iconify)
    glfwHideWindow(getRenderWindow());
#endif
#endif
#endif
}

void AsyGLRender::reshape(int width, int height)
{
  // Call base class to handle dimension updates and projection
  AsyRender::reshape(width, height);

  // OpenGL-specific: update viewport and mark SSBO for reinitialization
#ifdef HAVE_RENDERER
  glViewport(0, 0, Width, Height);
  if(ssbo)
    initSSBO = true;
#endif
}

} // namespace camp

#endif // HAVE_LIBGLM

#endif // HAVE_RENDERER
