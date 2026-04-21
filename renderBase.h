#pragma once

#include "common.h"

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
#include <functional>

#ifdef HAVE_PTHREAD
#include <pthread.h>
#endif

#include "common.h"
#include "ThreadSafeQueue.h"

#include "glmCommon.h"
#include "material.h"
#include "pen.h"
#include "triple.h"
#include "seconds.h"
#include "statistics.h"
#include "render.h"

namespace camp
{

class picture;
struct drawElement;

#define EMPTY_VIEW 0, nullptr
#define SINGLETON_VIEW(x) 1, &(x)
#define VEC_VIEW(x) static_cast<uint32_t>((x).size()), (x).data()
#define STD_ARR_VIEW(x) static_cast<uint32_t>((x).size()), (x).data()
#define ARR_VIEW(x) static_cast<uint32_t>(sizeof(x) / sizeof((x)[0])), x
#define RAW_VIEW(x) static_cast<uint32_t>(sizeof(x)), x
#define ST_VIEW(s) static_cast<uint32_t>(sizeof(s)), &s

template<class T>
inline T ceilquotient(T a, T b)
{
  return (a + b - 1) / b;
}

// Runtime error handling (used by both OpenGL and Vulkan)
inline void runtimeError(const std::string& s)
{
  cerr << "error: " << s << endl;
  exit(-1);
}

// Utility functions for power-of-two checks (used by both OpenGL and Vulkan)
inline bool ispow2(unsigned int n) {return n > 0 && !(n & (n - 1));}
inline void checkpow2(unsigned int n, std::string s) {
  if(!ispow2(n)) {
    runtimeError(s+" must be a power of two");
    exit(-1);
  }
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

enum DrawMode: int
{
   DRAWMODE_NORMAL,
   DRAWMODE_OUTLINE,
   DRAWMODE_WIREFRAME
};

constexpr int NUM_DRAW_MODES = 3;

struct Light
{
  glm::vec4 direction;
  glm::vec4 color;
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

/**
 * RendererMessage - Message types for inter-thread communication.
 * Used to send commands from asymain thread to render thread.
 */
enum class RendererMessage
{
  exportRender,   // Request to export/render a frame
  updateRenderer  // Request to update renderer state
};

/**
 * AsyRender - Library-agnostic base class for renderers.
 * Contains code that is independent of the underlying graphics API (Vulkan, OpenGL, etc.).
 */
class AsyRender
{
public:
  AsyRender() = default;
  virtual ~AsyRender() = default;

  /** Argument for render function - library-agnostic parameters */
  struct RenderFunctionArgs: public gc
  {
    string prefix;
    picture const* pic;
    string format;
    double width;
    double height;
    double angle;
    double zoom;
    triple m;
    triple M;
    pair shift;
    pair margin;

    double* t;
    double* tup;
    double* background;

    size_t nlightsin;

    triple* lights;
    double* diffuse;
    double* specular;

    bool view;
    int oldpid=0;
  };

  /** Pure virtual function that derived classes must implement */
  virtual void render(RenderFunctionArgs const& args) = 0;

  double getRenderResolution(triple Min) const;

  // Scene bounds
  double Xmin, Xmax;
  double Ymin, Ymax;
  double Zmin, Zmax;

  // Projection and camera state
  bool orthographic;
  glm::dmat4 rotateMat;
  glm::dmat4 projMat;
  glm::dmat4 viewMat;
  glm::dmat4 projViewMat;  // Combined projection*view matrix for offscreen culling

  // Viewport dimensions
  int fullWidth, fullHeight;
  double X, Y;
  double Angle;
  double Zoom;
  double Zoom0;
  pair Shift;
  pair Margin;
  double ArcballFactor;

  // Lighting
  camp::triple* Lights;
  double* LightsDiffuse;
  size_t nlights;
  std::array<float, 4> Background;

  // Viewport/clip bounds
  double xmin, xmax;
  double ymin, ymax;

  // Materials
  std::vector<Material> materials;
  MaterialMap materialMap;
  bool Opaque;

  // Draw mode
  DrawMode mode = DRAWMODE_NORMAL;

  // Window/viewport management
  int screenWidth, screenHeight;
  int Width, Height;
  int oldWidth, oldHeight;
  double Aspect;
  double oWidth, oHeight;
  double lastzoom;
  int Fitscreen=1;

  bool readyForExport=false;
  bool readyAfterExport=false;

  // Child process ID for export (used by both OpenGL and Vulkan)
  int Oldpid = 0;

#ifdef HAVE_RENDERER
  // GLFW window pointer (shared between Vulkan and OpenGL renderers)
  // Using void* to avoid requiring GLFW header in all files that include this
  void* glfwWindow = nullptr;

  /** Returns the GLFW window pointer as void*. Cast to appropriate type by caller. */
  void* getGLFWWindow() const { return glfwWindow; }
#endif

  // Timer and statistics
  utils::stopWatch spinTimer;
  utils::stopWatch fpsTimer;
  utils::stopWatch frameTimer;
  utils::statistics fpsStats;
  size_t framecount = 0;

  // Rendering state flags
  bool redraw=false;
  bool redisplay=false;
  bool resize=false;
  bool remesh=true;
  bool antialias = false;
  bool ibl = false;
  bool queueExport=false;
  bool haveScene=false;
  bool waitEvent=true;

  // Window state (used by mainLoop)
  bool havewindow = false;

  // Initialization state (used by both OpenGL and Vulkan)
  bool initialized = false;
  bool copied = false;
  bool format3dWait = false;

  // SSBO support flag (true if shader storage buffers are available)
  // Vulkan always supports SSBOs, OpenGL may or may not - default to false for safety
  bool ssbo = false;  // Will be set based on capability detection
  bool interlock = false;  // Fragment shader interlock support

  // Renderer-specific initialization flags (used by both OpenGL and Vulkan)
  bool initSSBO = true;  // Flag to indicate if SSBO buffers need initialization
  bool firstFit = true;  // Flag for initial fit screen adjustment
  bool ViewExport = false;  // Whether to export during view

  bool thread=false;

  // Window visibility
  bool hideWindow=false;

  // Spin state
  std::function<void()> currentIdleFunc = nullptr;
  bool Xspin = false;
  bool Yspin = false;
  bool Zspin = false;

  // Picture reference
  const picture* pic = nullptr;

  // Mouse/interaction state
  std::string lastAction = "";

  // Window title
  std::string title = "";

  // Internal state
  double H;
  double Xfactor, Yfactor;
  double cx, cy;

  // Transform matrices (used by camera())
  double T[16];
  double Tup[16];

  // Normal matrices for shader transformations
  glm::dmat3 normMat;  // Double precision normal matrix for CPU calculations

#ifdef HAVE_PTHREAD
  // Pthread synchronization primitives (shared between renderers)
  pthread_t mainthread;
  pthread_cond_t initSignal = PTHREAD_COND_INITIALIZER;
  pthread_mutex_t initLock = PTHREAD_MUTEX_INITIALIZER;
  pthread_cond_t readySignal = PTHREAD_COND_INITIALIZER;
  pthread_mutex_t readyLock = PTHREAD_MUTEX_INITIALIZER;

  // Message queue for inter-thread communication (asymain -> render thread)
  ThreadSafeQueue<RendererMessage> messageQueue;
#endif

protected:
  const double pi=acos(-1.0);
  const double degrees=180.0/pi;
  const double radians=1.0/degrees;

public:
  // String state (public for external access)
  string Format;
  bool View = false;
  string Prefix;

protected:
  // Virtual functions for derived classes to override (not pure - have default implementations)
  virtual void setDimensions(int Width, int Height, double X, double Y);
  virtual void updateModelViewData();

public:
  virtual void setProjection();

  // Update projection and view matrices
  virtual void update();

  // Projection matrix functions
  void updateProjection();
  virtual void frustum(double left, double right, double bottom,
                       double top, double nearVal, double farVal);
  virtual void ortho(double left, double right, double bottom,
                     double top, double nearVal, double farVal);

  // Clear functions (public for external access)
  void clearCenters();
  void clearMaterials();
  void clearData();
  void prepareScene();

  // Camera and view manipulation
  projection camera(bool user=true);
  void showCamera();

  // User interaction handlers
  void shift(double dx, double dy);
  void pan(double dx, double dy);
  void capzoom();
  void zoom(double dx, double dy);

  // Window size management
  void capsize(int& w, int& h);
  void windowposition(int& x, int& y, int width=-1, int height=-1);
  virtual void setsize(int w, int h, bool reposition=true);
  virtual void fullscreen(bool reposition=true);
  virtual void reshape(int width, int height);
  void setosize();
  virtual void fitscreen(bool reposition=true);
  virtual void toggleFitScreen();
  virtual void home(bool webgl=false);

  // Mode cycling
  virtual void cycleMode();

  // Spin controls
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

  // Idle function
  void idleFunc(std::function<void()> f);
  void idle();

  // Window management (library-agnostic parts)
  virtual void expand();
  virtual void shrink();

  virtual void updateHandler(int=0);
  virtual void exportHandler(int=0) = 0;
  virtual void Export(int imageIndex=0) = 0;

  // Message processing for inter-thread communication
  void processMessages(RendererMessage const& msg);

  // Display/render the current frame (library-agnostic implementation in base class)
  virtual void display();

  // Virtual hooks for library-specific operations (override in derived classes)
  virtual void swapBuffers();
  virtual void showWindow();
  virtual void drawFrame() = 0;

  virtual void quit();

  // Window close handler (library-agnostic)
  virtual void onClose();

  // Finalize graphics library resources (virtual for renderer-specific cleanup)
  virtual void finalizeProcess();

  // Key handling (library-agnostic)
  virtual void onKey(int key, int scancode, int action, int mods);

  // Main event loop (shared between OpenGL and Vulkan renderers)
  void mainLoop();

#ifdef HAVE_PTHREAD
  // Pthread synchronization helpers
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
};

// Renderer pointer - unified interface for both OpenGL and Vulkan
// This allows dynamic loading of the appropriate renderer at runtime
extern AsyRender* gl;  // Global renderer instance (type depends on build configuration)

#ifdef HAVE_RENDERER
class AsyGLRender;  // Forward declaration
class AsyVkRender;   // Forward declaration (if Vulkan is available)
#endif

void mode();

} // namespace camp
