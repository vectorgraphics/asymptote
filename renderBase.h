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

#ifdef HAVE_LIBGLFW
#ifndef GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_NONE
#endif
#include "glfw.h"
#endif

// GLM-independent utility functions (available even when HAVE_LIBGLM is undefined)
// These are used by both OpenGL and Vulkan renderers but don't depend on GLM.
namespace camp {

// Runtime error handling (used by both OpenGL and Vulkan)
inline void runtimeError(const std::string& s)
{
  cerr << "error: " << s << endl;
  exit(-1);
}

template<class T>
inline T ceilquotient(T a, T b)
{
  return (a + b - 1) / b;
}

// Return the smallest power of 2 greater than or equal to n.
inline unsigned int ceilpow2(unsigned int n)
{
  --n;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  return ++n;
}

// Utility functions for power-of-two checks (used by both OpenGL and Vulkan)
inline bool ispow2(unsigned int n) {return n > 0 && !(n & (n - 1));}
inline void checkpow2(unsigned int n, std::string s) {
  if(!ispow2(n)) {
    runtimeError(s+" must be a power of two");
    exit(-1);
  }
}

/**
 * Detect NVIDIA RTX 3000-series (Ampere) GPUs which have a known driver bug
 * with fragment shader interlock causing rendering artifacts.
 * Checks for "RTX 30", "RTX30", or chip codename "GA10" in the device string.
 */
inline bool isNVIDIA30xx(const char* deviceStr)
{
  if (!deviceStr) return false;
  string s(deviceStr);
  return s.find("NVIDIA") != string::npos &&
         (s.find("RTX 30") != string::npos ||
          s.find("RTX30") != string::npos ||
          s.find("GA10") != string::npos);
}

} // namespace camp

#ifdef HAVE_LIBGLM

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
class drawElement;

#define EMPTY_VIEW 0, nullptr
#define SINGLETON_VIEW(x) 1, &(x)
#define VEC_VIEW(x) static_cast<uint32_t>((x).size()), (x).data()
#define STD_ARR_VIEW(x) static_cast<uint32_t>((x).size()), (x).data()
#define ARR_VIEW(x) static_cast<uint32_t>(sizeof(x) / sizeof((x)[0])), x
#define RAW_VIEW(x) static_cast<uint32_t>(sizeof(x)), x
#define ST_VIEW(s) static_cast<uint32_t>(sizeof(s)), &s

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

// Verbosity threshold for timing partial sums (GPU indexing benchmarking)
constexpr Int timePartialSumVerbosity = 4;

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
    static triple Z(0.0, 0.0, 1.0);
    axis = unit(cross(v0, v1), Z);
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
  updateRenderer, // Request to update renderer state
  createRenderer  // Request the render thread to load and create the renderer
};

#ifdef HAVE_PTHREAD
/**
 * ThreadManager - Manages thread synchronization primitives.
 *
 * This class owns all pthread condition variables, mutexes, and the message
 * queue used for inter-thread communication between the asymain thread and
 * the render thread. It is independent of any specific rendering backend
 * (OpenGL, Vulkan, etc.).
 */
class ThreadManager
{
public:
  ThreadManager() = default;
  ~ThreadManager() = default;

  // Non-copyable, non-movable
  ThreadManager(const ThreadManager&) = delete;
  ThreadManager& operator=(const ThreadManager&) = delete;

  /// Main thread identifier (set by the loader when threading is enabled)
  pthread_t mainthread;

  /// Initialization signaling: render thread waits for asymain to finish setup
  pthread_cond_t initSignal = PTHREAD_COND_INITIALIZER;
  pthread_mutex_t initLock = PTHREAD_MUTEX_INITIALIZER;

  /// Export readiness signaling: asymain waits for render thread after export
  pthread_cond_t readySignal = PTHREAD_COND_INITIALIZER;
  pthread_mutex_t readyLock = PTHREAD_MUTEX_INITIALIZER;

  /// Message queue for inter-thread communication (asymain -> render thread)
  ThreadSafeQueue<RendererMessage> messageQueue;

  /// Signaling for renderer creation: asymain waits for render thread to finish creating
  pthread_cond_t createdSignal = PTHREAD_COND_INITIALIZER;
  pthread_mutex_t createdLock = PTHREAD_MUTEX_INITIALIZER;

  /**
   * Signal a condition variable and release its mutex.
   * Used to notify another thread that an operation has completed.
   */
  void endwait(pthread_cond_t& signal, pthread_mutex_t& lock)
  {
    pthread_mutex_lock(&lock);
    pthread_cond_signal(&signal);
    pthread_mutex_unlock(&lock);
  }

  /**
   * Signal a condition variable and then wait on it.
   * Used for handshaking: signal the other thread, then block until
   * the other thread signals back.
   */
  void wait(pthread_cond_t& signal, pthread_mutex_t& lock)
  {
    pthread_mutex_lock(&lock);
    pthread_cond_signal(&signal);
    pthread_cond_wait(&signal, &lock);
    pthread_mutex_unlock(&lock);
  }
};
#endif // HAVE_PTHREAD

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

    bool view;
    int oldpid=0;
  };

  /** Entry point for rendering. Called by the asymain thread to process
   * a render request. Derived classes must initialize their backend, copy
   * arguments via copyRenderArgs(), and either enter mainLoop() for interactive
   * viewing or produce output for export. */
  virtual void render(RenderFunctionArgs const& args) = 0;

  /** Copy common arguments from RenderFunctionArgs to member variables.
   * Shared by all renderer implementations (Vulkan, OpenGL, WebGL). */
  void copyRenderArgs(RenderFunctionArgs const& args);

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
  double devicePixelRatio;
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

  // GLFW window pointer (shared between Vulkan and OpenGL renderers)
#ifdef HAVE_LIBGLFW
  GLFWwindow* glfwWindow = nullptr;
#else
  void* glfwWindow = nullptr;
#endif

  /** Returns the GLFW window pointer. Cast to GLFWwindow* when HAVE_LIBGLFW is defined. */
  void* getGLFWWindow() const { return glfwWindow; }

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

  // Spin state
  static inline bool threads = false;

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
  double xprev = 0.0;
  double yprev = 0.0;
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

  /// Accessor for combined projection*view matrix (member version of getProjViewMat())
  const glm::dmat4& getProjViewMat() const { return projViewMat; }
  /// Accessor for view matrix (member version of getViewMat())
  const glm::dmat4& getViewMat() const     { return viewMat; }
  /// Accessor for normal matrix (member version of getNormMat())
  const glm::dmat3& getNormMat() const     { return normMat; }

#ifdef HAVE_PTHREAD
  // Thread synchronization manager (shared between renderers)
  ThreadManager threadMgr;
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
  /** Set viewport dimensions and content offset. Called during initialization
   * and after resize. Derived classes may override to adjust backend-specific
   * state (e.g., Vulkan swapchain extents). */
  virtual void setDimensions(int Width, int Height, double X, double Y);

  /** Update model-view uniform data for the current camera state.
   * Called before drawing when matrices have changed. */
  virtual void updateModelViewData();

public:
  /** Compute and set the projection matrix from current orthographic/perspective
   * state, aspect ratio, and zoom. Derived classes may override to upload the
   * resulting matrix to a GPU uniform buffer. */
  virtual void setProjection();

  /** Update all transformation matrices (view, projection, combined proj*view).
   * Called each frame before drawing. Vulkan overrides to also set a dirty flag
   * for uniform buffer upload. */
  virtual void update();

  /** Set the projection matrix to a perspective frustum.
   * Derived classes may override for backend-specific handling. */
  virtual void frustum(double left, double right, double bottom,
                       double top, double nearVal, double farVal);

  /** Set the projection matrix to an orthographic frustum.
   * Derived classes may override for backend-specific handling. */
  virtual void ortho(double left, double right, double bottom,
                     double top, double nearVal, double farVal);

  /** Recompute the projection matrix and notify derived classes to upload.
   * Called after setProjection() or when viewport changes. */
  void updateProjection();

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
  /// Given max framebuffer dimensions, return the largest size that fits
  /// while preserving the content Aspect ratio.
  void fitAspect(int& w, int& h);
  void windowposition(int& x, int& y, int width=-1, int height=-1);
  /** Set the window size and optionally reposition. Derived classes may override
   * to update backend-specific resources (e.g., Vulkan swapchain recreation). */
  virtual void setsize(int w, int h, bool reposition=true);

  /** Toggle fullscreen mode. Derived classes may override for platform-specific
   * window management. */
  virtual void fullscreen(bool reposition=true);

  /** Handle framebuffer reshape. Called when the window is resized.
   * Derived classes may override to update viewport or swapchain state. */
  virtual void reshape(int width, int height);
  void setosize();
  /** Fit the window to screen dimensions, preserving aspect ratio.
   * Derived classes may override for backend-specific behavior. */
  virtual void fitscreen(bool reposition=true);

  /** Toggle fit-screen mode on/off. */
  virtual void toggleFitScreen();

  void initDisplay(int contentW, int contentH);
  void setOpaque();
  /** Reset the camera to its default home position. */
  virtual void home();

  /** Cycle through draw modes (normal, outline, wireframe).
   * Derived classes may override to rebuild pipelines or shaders. */
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
  /** Expand the window by increasing its dimensions. */
  virtual void expand();

  /** Shrink the window by decreasing its dimensions. */
  virtual void shrink();

  /** Handle an update request from the asymain thread.
   * Derived classes may override to refresh backend state (e.g., rebuild shaders). */
  virtual void updateHandler(int=0);

  /** Handle an export request. The base class sets readyAfterExport;
   * derived classes override to perform backend-specific export setup. */
  virtual void exportHandler(int);

  /** Export the current frame to an image or file. Pure virtual -- each
   * backend must implement its own pixel readback and encoding. */
  virtual void Export(int imageIndex=0) = 0;

  // Message processing for inter-thread communication
  void processMessages(RendererMessage const& msg);

  /** Main display loop body. Calls prepareScene(), showWindow(), drawFrame(),
   * and swapBuffers() in sequence, with FPS tracking and message processing.
   * Derived classes typically do not override this; instead they override the
   * individual virtual hooks called by this method. */
  virtual void display();

  /** Swap front and back buffers. Called by display() after drawFrame().
   * For Vulkan, presentation is handled inside drawFrame() via presentKHR,
   * so the default implementation is a no-op. OpenGL overrides to call
   * glfwSwapBuffers. */
  virtual void swapBuffers();

  /** Show the render window if it is hidden. Called by display() before
   * drawFrame(). Derived classes implement using GLFW show/hide calls. */
  virtual void showWindow();

  /** Render a single frame. Called by display() after prepareScene() and
   * showWindow(). The viewport is already set, scene data (materials,
   * triangles, etc.) is populated, and Width/Height reflect the current
   * framebuffer dimensions. Pure virtual -- each backend must implement. */
  virtual void drawFrame() = 0;

  /** Request termination of the render loop. */
  virtual void quit();

  /** Handle window close request. Called by the GLFW close callback.
   * Derived classes may override for backend-specific cleanup. */
  virtual void onClose();

  /** Release all graphics library resources (buffers, shaders, pipelines).
   * Called during shutdown. Derived classes must implement backend-specific
   * resource destruction. */
  virtual void finalizeProcess();

  /** Handle key press/release events. Called by GLFW keyboard callback.
   * The base class implements shared key bindings (quit, fullscreen, mode
   * cycling, spin). Derived classes may override for additional handling. */
  virtual void onKey(int key, int scancode, int action, int mods);

  /** Handle scroll wheel events. Called by GLFW scroll callback.
   * The base class maps scroll to zoom. Derived classes may override. */
  virtual void onScroll(double xoffset, double yoffset);

  /** Handle mouse button press/release. Called by GLFW mouse button callback.
   * The base class tracks click position and action for pan/rotate.
   * Derived classes may override (e.g., Vulkan adds focus loss handling). */
  virtual void onMouseButton(int button, int action, int mods);

  /** Handle cursor position changes. Called by GLFW cursor pos callback.
   * The base class computes delta from previous position for pan/rotate.
   * Derived classes may override for additional tracking. */
  virtual void onCursorPos(double xpos, double ypos);

  /** Handle framebuffer resize. Called by GLFW framebuffer resize callback.
   * Derived classes typically override to update viewport and recreate
   * backend-specific resources (e.g., Vulkan swapchain/images). */
  virtual void onFramebufferResize(int width, int height);

  // Main event loop (shared between OpenGL and Vulkan renderers)
  void mainLoop();

#ifdef HAVE_PTHREAD
  // Pthread synchronization helpers (forward to ThreadManager)
  void endwait(pthread_cond_t& signal, pthread_mutex_t& lock)
  {
    threadMgr.endwait(signal, lock);
  }
  void wait(pthread_cond_t& signal, pthread_mutex_t& lock)
  {
    threadMgr.wait(signal, lock);
  }
#endif
};

// Renderer pointer - unified interface for both OpenGL and Vulkan
// This allows dynamic loading of the appropriate renderer at runtime
extern AsyRender* gl;  // Global renderer instance (type depends on build configuration)

#ifdef HAVE_RENDERER
#ifdef HAVE_GL
class AsyGLRender;  // Forward declaration
#endif
#ifdef HAVE_VULKAN
class AsyVkRender;   // Forward declaration (if Vulkan is available)
#endif
class NoRender;  // Forward declaration for WebGL/v3d output

/**
 * Lazily initialise the global renderer (Vulkan, OpenGL, or NoRender) on first use.
 * This defers all graphics-library loading until a shipout3 call actually
 * requires rendering, allowing headless modes like "-l" to run without
 * needing any GPU / display at all.
 *
 * @param format Output format string (e.g., "html", "v3d", or nullptr for default)
 */
#endif

void initRenderer(const char* format);

void mode();

} // namespace camp
#endif // HAVE_LIBGLM
