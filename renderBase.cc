#include "renderBase.h"
#include "settings.h"
#include "drawelement.h"
#include "interact.h"
#include "picture.h"

#ifdef HAVE_RENDERER
// Forward declaration for GLFWwindow to avoid including glfw3.h here
struct GLFWwindow;
#endif

using settings::getSetting;
using namespace glm;

// Forward declaration for pthread callback (defined in glfw.cc)
void *postEmptyEvent(void *);

namespace camp
{

double AsyRender::getRenderResolution(triple Min) const
{
  double prerender = settings::getSetting<double>("prerender");

  if (prerender <= 0.0)
    return 0.0;

  prerender = 1.0 / prerender;
  double perspective = orthographic || Zmax == 0.0 ? 0.0 : 1.0 / Zmax;
  double s = perspective ? Min.getz() * perspective : 1.0;
  triple b(Xmin, Ymin, Zmin);
  triple B(Xmax, Ymax, Zmax);
  pair size3(s * (B.getx() - b.getx()), s * (B.gety() - b.gety()));
  pair size2(Width, Height);
  return prerender * size3.length() / size2.length();
}

// Default implementations for virtual methods that can have generic behavior
void AsyRender::setDimensions(int Width, int Height, double X, double Y)
{
  double aspect = ((double) Width) / Height;
  double xshift = (X / (double) Width + Shift.getx() * Xfactor) * Zoom;
  double yshift = (Y / (double) Height + Shift.gety() * Yfactor) * Zoom;
  double zoominv = 1.0 / Zoom;
  if (orthographic) {
    double xsize = Xmax - Xmin;
    double ysize = Ymax - Ymin;
    if (xsize < ysize * aspect) {
      double r = 0.5 * ysize * aspect * zoominv;
      double X0 = 2.0 * r * xshift;
      double Y0 = ysize * zoominv * yshift;
      xmin = -r - X0;
      xmax = r - X0;
      ymin = Ymin * zoominv - Y0;
      ymax = Ymax * zoominv - Y0;
    } else {
      double r = 0.5 * xsize * zoominv / aspect;
      double X0 = xsize * zoominv * xshift;
      double Y0 = 2.0 * r * yshift;
      xmin = Xmin * zoominv - X0;
      xmax = Xmax * zoominv - X0;
      ymin = -r - Y0;
      ymax = r - Y0;
    }
  } else {
    double r = H * zoominv;
    double rAspect = r * aspect;
    double X0 = 2.0 * rAspect * xshift;
    double Y0 = 2.0 * r * yshift;
    xmin = -rAspect - X0;
    xmax = rAspect - X0;
    ymin = -r - Y0;
    ymax = r - Y0;
  }
}

void AsyRender::setProjection()
{
  setDimensions(Width, Height, X, Y);

  if(haveScene) {
    if(orthographic) ortho(xmin,xmax,ymin,ymax,-Zmax,-Zmin);
    else frustum(xmin,xmax,ymin,ymax,-Zmax,-Zmin);
  }
}

void AsyRender::updateModelViewData()
{
  // Update normal matrix (inverse transpose of view matrix rotation)
  normMat = dmat3(inverse(viewMat));
}

void AsyRender::update()
{
  capzoom();

  double cz = 0.5 * (Zmin + Zmax);
  viewMat = translate(translate(dmat4(1.0), dvec3(cx, cy, cz)) * rotateMat, dvec3(0, 0, -cz));

  setProjection();
  updateModelViewData();

  redraw=true;

#ifdef HAVE_PTHREAD
  if(View) {
    pthread_t postThread;
    if(pthread_create(&postThread,NULL,postEmptyEvent,NULL) == 0)
      pthread_join(postThread,NULL);
  }
#endif
}

void AsyRender::updateProjection()
{
  projViewMat = projMat * viewMat;
}

void AsyRender::frustum(double left, double right, double bottom,
                        double top, double nearVal, double farVal)
{
  projMat = glm::frustum(left, right, bottom, top, nearVal, farVal);
  updateProjection();
}

void AsyRender::ortho(double left, double right, double bottom,
                      double top, double nearVal, double farVal)
{
  projMat = glm::ortho(left, right, bottom, top, nearVal, farVal);
  updateProjection();
}

void AsyRender::clearCenters()
{
  drawElement::centers.clear();
  drawElement::centermap.clear();
}

void AsyRender::clearMaterials()
{
  materials.clear();
  materialMap.clear();
}

void AsyRender::clearData()
{
  pointData.clear();
  lineData.clear();
  materialData.clear();
  colorData.clear();
  triangleData.clear();
  transparentData.clear();
}

void AsyRender::prepareScene()
{

#ifdef HAVE_PTHREAD
  static bool first=true;
  if(thread && first) {
    wait(initSignal,initLock);
    endwait(initSignal,initLock);
    first=false;
  }

  if(format3dWait)
    wait(initSignal,initLock);
#endif

  if(redraw) {
    clearData();

    if(remesh)
      clearCenters();

    triple m(xmin,ymin,Zmin);
    triple M(xmax,ymax,Zmax);
    double perspective=orthographic || Zmax == 0.0 ? 0.0 : 1.0/Zmax;

    double size2=hypot(Width,Height);

    pic->render(size2,m,M,perspective,remesh);
    redraw=false;

    if(mode != DRAWMODE_OUTLINE)
      remesh=false;

    Opaque=transparentData.indices.empty();
  }
}

projection AsyRender::camera(bool user)
{
  camp::Triple vCamera, vUp, vTarget;

  double cz = 0.5 * (Zmin + Zmax);

  double *Rotate = value_ptr(rotateMat);

  if(user) {
    double shift[]={0.0,0.0,0.0,0.0};
    for(int i=0; i < 3; ++i) {
      double sumCamera=0.0, sumTarget=0.0, sumUp=0.0;
      int i4=4*i;
      shift[3]=T[i4+2]*cz;
      for(int j=0; j < 4; ++j) {
        int j4=4*j;
        double R0=Rotate[j4];
        double R1=Rotate[j4+1];
        double R2=Rotate[j4+2];
        double R3=Rotate[j4+3];
        double T4ij=T[i4+j]+shift[j]; // T -> T*shift(0,0,cz);
        sumCamera += T4ij*(R3-cx*R0-cy*R1-cz*R2);
        sumUp += Tup[i4+j]*R1;
        sumTarget += T4ij*(R3-cx*R0-cy*R1);
      }
      vCamera[i]=sumCamera;
      vUp[i]=sumUp;
      vTarget[i]=sumTarget;
    }
  } else {
    for(int i=0; i < 3; ++i) {
      int i4=4*i;
      double R0=Rotate[i4];
      double R1=Rotate[i4+1];
      double R2=Rotate[i4+2];
      double R3=Rotate[i4+3];
      vCamera[i]=R3-cx*R0-cy*R1-cz*R2;
      vUp[i]=R1;
      vTarget[i]=R3-cx*R0-cy*R1;
    }
  }

  return projection(orthographic, vCamera, vUp, vTarget, Zoom,
                    2.0*atan(tan(0.5*Angle)/Zoom)/radians,
                    pair(X/Width+Shift.getx(),
                         Y/Height+Shift.gety()));
}

void AsyRender::showCamera()
{
  projection P = camera();
  string projectionStr = P.orthographic ? "orthographic(" : "perspective(";
  string indent(2 + projectionStr.length(), ' ');
  cout << endl
       << "currentprojection=" << endl << "  "
       << projectionStr << "camera=" << P.camera << "," << endl
       << indent << "up=" << P.up << "," << endl
       << indent << "target=" << P.target << "," << endl
       << indent << "zoom=" << P.zoom;
  if(!orthographic)
    cout << "," << endl << indent << "angle=" << P.angle;
  if(P.viewportshift != pair(0.0,0.0))
    cout << "," << endl << indent << "viewportshift=" << P.viewportshift*Zoom;
  if(!orthographic)
    cout << "," << endl << indent << "autoadjust=false";
  cout << ");" << endl;
}

void AsyRender::shift(double dx, double dy)
{
  double Zoominv = 1.0 / Zoom;

  X += dx * Zoominv;
  Y += -dy * Zoominv;
  update();
}

void AsyRender::pan(double dx, double dy)
{
  if(orthographic)
    shift(dx, dy);
  else {
    cx += dx * (xmax - xmin) / Width;
    cy -= dy * (ymax - ymin) / Height;
    update();
  }
}

void AsyRender::capzoom()
{
  static double maxzoom = sqrt(DBL_MAX);
  static double minzoom = 1.0 / maxzoom;
  if(Zoom <= minzoom) Zoom = minzoom;
  if(Zoom >= maxzoom) Zoom = maxzoom;

  if(fabs(Zoom - lastzoom) > settings::getSetting<double>("zoomThreshold")) {
    remesh = true;
    lastzoom = Zoom;
  }
}

void AsyRender::zoom(double dx, double dy)
{
  double zoomFactor = settings::getSetting<double>("zoomfactor");
  if (zoomFactor > 0.0) {
    double zoomStep = settings::getSetting<double>("zoomstep");
    const double limit = log(0.1*DBL_MAX) / log(zoomFactor);
    double stepPower = zoomStep * dy;
    if(fabs(stepPower) < limit) {
      Zoom *= std::pow(zoomFactor, -stepPower);
      update();
    }
  }
}

void AsyRender::capsize(int& width, int& height)
{
  if(width > screenWidth)
    width = screenWidth;
  if(height > screenHeight)
    height = screenHeight;
}

void AsyRender::windowposition(int& x, int& y, int width, int height)
{
  if (width == -1) {
    width = Width;
  }
  if (height == -1) {
    height = Height;
  }

  pair z = settings::getSetting<pair>("position");
  x = (int) z.getx();
  y = (int) z.gety();
  if(x < 0) {
    x += screenWidth - width;
    if(x < 0) x = 0;
  }
  if(y < 0) {
    y += screenHeight - height;
    if(y < 0) y = 0;
  }
}

void AsyRender::setosize()
{
  oldWidth = (int) ceil(oWidth);
  oldHeight = (int) ceil(oHeight);
}

/**
 * Set window to fullscreen size.
 * Base implementation handles dimension calculation and GLFW window operations.
 */
void AsyRender::fullscreen(bool reposition)
{
  Width = screenWidth;
  Height = screenHeight;
  Xfactor = ((double) screenHeight) / Height;
  Yfactor = ((double) screenWidth) / Width;

  setsize(Width, Height, reposition);
}

/**
 * Set window size and optionally reposition.
 * Base implementation handles GLFW window operations and common logic.
 */
void AsyRender::setsize(int w, int h, bool reposition)
{
#ifdef HAVE_RENDERER
  // Handle GLFW window operations (library-agnostic for Vulkan/OpenGL)
  if (View && glfwWindow != nullptr) {
    ::glfwSetWindowSize(static_cast<GLFWwindow*>(glfwWindow), w, h);
    if (reposition) {
      int x, y;
      windowposition(x, y, w, h);
      ::glfwSetWindowPos(static_cast<GLFWwindow*>(glfwWindow), x, y);
    }
  }
#endif

  capsize(w, h);
  reshape(w, h);
  update();
}

/**
 * Handle window resize.
 * Base implementation handles dimension updates and projection.
 */
void AsyRender::reshape(int width, int height)
{
  // Scale X,Y proportionally with new dimensions
  X = (X / Width) * width;
  Y = (Y / Height) * height;

  Width = width;
  Height = height;

  static int lastWidth = 1;
  static int lastHeight = 1;
  if (View && width * height > 1 &&
      (width != lastWidth || height != lastHeight)) {

    if (settings::verbose > 1)
      cout << "Rendering " << stripDir(Prefix) << " as "
           << width << "x" << height << " image" << endl;
    lastWidth = width;
    lastHeight = height;
  }

  setProjection();
}

void AsyRender::fitscreen(bool reposition)
{
  switch(Fitscreen) {
    case 0: // Original size
    {
      Xfactor = Yfactor = 1.0;
      double pixelRatio = settings::getSetting<double>("devicepixelratio");
      setsize(oldWidth*pixelRatio, oldHeight*pixelRatio, reposition);
      break;
    }
    case 1: // Fit to screen in one dimension
    {
      int w = screenWidth;
      int h = screenHeight;
      if(w > h * Aspect)
        w = min((int) ceil(h * Aspect), w);
      else
        h = min((int) ceil(w / Aspect), h);

      setsize(w, h, reposition);
      break;
    }
    case 2: // Full screen
    {
      fullscreen(reposition);
      break;
    }
  }
}

void AsyRender::toggleFitScreen()
{
#ifdef HAVE_RENDERER
  // Hide window before changing size (library-agnostic for Vulkan/OpenGL)
  if (glfwWindow != nullptr) {
    ::glfwHideWindow(static_cast<GLFWwindow*>(glfwWindow));
  }
#endif

  Fitscreen = (Fitscreen + 1) % 3;
  fitscreen();
}

void AsyRender::home(bool webgl)
{
  if(!webgl)
    idle();
  X = Y = cx = cy = 0;
  rotateMat = viewMat = dmat4(1.0);
  lastzoom = Zoom = Zoom0;
  framecount = 0;

  setProjection();
  updateModelViewData();
}

void AsyRender::cycleMode()
{
  mode = DrawMode((mode + 1) % NUM_DRAW_MODES);
  remesh = true;
  redraw = true;

  // Update IBL setting based on mode
  if (mode == DRAWMODE_NORMAL) {
    ibl = settings::getSetting<bool>("ibl");
  } else if (mode == DRAWMODE_OUTLINE) {
    ibl = false;
  }
}

double AsyRender::spinStep()
{
  return settings::getSetting<double>("spinstep") * spinTimer.seconds(true);
}

void AsyRender::rotateX(double step)
{
  dmat4 tmpRot(1.0);
  tmpRot = rotate(tmpRot, glm::radians(step), dvec3(1, 0, 0));
  rotateMat = tmpRot * rotateMat;

  update();
}

void AsyRender::rotateY(double step)
{
  dmat4 tmpRot(1.0);
  tmpRot = rotate(tmpRot, glm::radians(step), dvec3(0, 1, 0));
  rotateMat = tmpRot * rotateMat;

  update();
}

void AsyRender::rotateZ(double step)
{
  dmat4 tmpRot(1.0);
  tmpRot = rotate(tmpRot, glm::radians(step), dvec3(0, 0, 1));
  rotateMat = tmpRot * rotateMat;

  update();
}

void AsyRender::xspin()
{
  rotateX(spinStep());
}

void AsyRender::yspin()
{
  rotateY(spinStep());
}

void AsyRender::zspin()
{
  rotateZ(spinStep());
}

void AsyRender::spinx()
{
  if(Xspin)
    idle();
  else {
    idleFunc([this](){xspin();});
    Xspin = true;
    Yspin = Zspin = false;
  }
}

void AsyRender::spiny()
{
  if(Yspin)
    idle();
  else {
    idleFunc([this](){yspin();});
    Yspin = true;
    Xspin = Zspin = false;
  }
}

void AsyRender::spinz()
{
  if(Zspin)
    idle();
  else {
    idleFunc([this](){zspin();});
    Zspin = true;
    Xspin = Yspin = false;
  }
}

void AsyRender::idleFunc(std::function<void()> f)
{
  spinTimer.reset();
  currentIdleFunc = f;
}

void AsyRender::idle()
{
  idleFunc(nullptr);
  Xspin = Yspin = Zspin = false;
}

void AsyRender::expand()
{
  double resizeStep = settings::getSetting<double>("resizestep");
  if(resizeStep > 0.0)
    setsize((int) (Width*resizeStep+0.5), (int) (Height*resizeStep+0.5));
}

void AsyRender::shrink()
{
  double resizeStep = settings::getSetting<double>("resizestep");
  if(resizeStep > 0.0)
    setsize(max((int) (Width/resizeStep+0.5),1),
            max((int) (Height/resizeStep+0.5),1));
}

void AsyRender::exportHandler(int)
{
  // Default implementation - derived classes should override
}

/**
 * Update handler - common to both OpenGL and Vulkan renderers.
 * Hides window if viewing interactively and sets rendering flags.
 */
void AsyRender::updateHandler(int)
{
#ifdef HAVE_RENDERER
  if(View && !interact::interactive) {
    ::glfwHideWindow(static_cast<GLFWwindow*>(getGLFWWindow()));
    if(!getSetting<bool>("fitscreen"))
      Fitscreen=0;
  }
#endif

  resize=true;
  redisplay=true;
  redraw=true;
  remesh=true;
  waitEvent=false;
}

/**
 * Process messages from the message queue (inter-thread communication).
 */
void AsyRender::processMessages(RendererMessage const& msg)
{
  switch (msg)
  {
    case RendererMessage::exportRender:
      if (readyForExport)
      {
        readyForExport=false;
        exportHandler(0);
      }
      break;
    case RendererMessage::updateRenderer:
      updateHandler(0);
      break;
    default:
      break;
  }
}

void AsyRender::quit()
{
#ifdef HAVE_RENDERER
  // Stop all rendering activity
  resize = false;
  waitEvent = false;
  redraw = false;

  if (thread) {
#ifdef HAVE_PTHREAD
    if (!interact::interactive) {
      idle();
      endwait(readySignal, readyLock);
    }
#endif

    // Hide window but don't destroy it (will be reused)
    if (View && glfwWindow) {
      ::glfwHideWindow(static_cast<GLFWwindow*>(glfwWindow));
      hideWindow = true;
    }
    // In threaded mode, don't call exit() - the main thread handles that
  } else {
    // Non-threaded mode: finalize graphics library before cleanup
    finalizeProcess();

    // Clean up and exit
    if (View && glfwWindow) {
      ::glfwDestroyWindow(static_cast<GLFWwindow*>(glfwWindow));
      glfwWindow = nullptr;
    }

    // Terminate GLFW before exiting
    glfwTerminate();

    exit(0);
  }
#endif
}

/**
 * Finalize graphics library resources.
 * Default implementation does nothing; derived classes override for specific cleanup.
 */
void AsyRender::finalizeProcess()
{
  // Default: no-op
}

/**
 * Display/render the current frame (library-agnostic implementation).
 * Uses virtual hooks for library-specific operations.
 */
void AsyRender::display()
{
  prepareScene();

  // Show window if needed (library-specific)
  showWindow();

  // Draw the frame (renderer-specific)
  drawFrame();

  // FPS tracking
  bool fps = settings::verbose > 2;
  if(fps) {
    if(framecount < 20) fpsTimer.reset();
    else {
      double s = fpsTimer.seconds(true);
      if(s > 0.0) {
        double rate = 1.0/s;
        fpsStats.add(rate);
        if(framecount % 20 == 0)
          cout << "FPS=" << rate << "\t" << fpsStats.mean()
               << " +/- " << fpsStats.stdev() << endl;
      }
    }
    ++framecount;
  }

  // Swap buffers (library-specific)
  swapBuffers();

  // Process management (non-Windows)
  if(!thread) {
#if defined(_WIN32)
#else
    if(Oldpid != 0 && waitpid(Oldpid, NULL, WNOHANG) != Oldpid) {
      kill(Oldpid, SIGHUP);
      Oldpid = 0;
    }
#endif
  }
}

/**
 * Swap front and back buffers (library-specific).
 * Default: no-op - override in derived classes.
 */
void AsyRender::swapBuffers()
{
  // Default: no-op - derived classes must override
}

/**
 * Show the window if hidden (library-specific).
 * Default: no-op - override in derived classes.
 */
void AsyRender::showWindow()
{
  // Default: no-op - derived classes can override for specific behavior
}

/**
 * Window close handler (library-agnostic).
 * Can be overridden by derived classes for renderer-specific cleanup.
 */
void AsyRender::onClose()
{
  // Default: no-op - derived classes can override for specific behavior
}

void AsyRender::onKey(int key, int scancode, int action, int mods)
{
    if (action != GLFW_PRESS)
        return;

    switch (key)
    {
        case 'H':
            home();
            redraw = true;
            break;
        case 'F':
            toggleFitScreen();
            break;
        case 'X':
            spinx();
            break;
        case 'Y':
            spiny();
            break;
        case 'Z':
            spinz();
            break;
        case 'S':
            idle();
            break;
        case 'M':
            cycleMode();
            break;
        case 'E':
            queueExport = true;
            break;
        case 'C':
            showCamera();
            break;
        case '.': // '>' = '.' + shift
            if (!(mods & GLFW_MOD_SHIFT))
                break;
        case '+':
        case '=':
            expand();
            break;
        case ',': // '<' = ',' + shift
            if (!(mods & GLFW_MOD_SHIFT))
                break;
        case '-':
        case '_':
            shrink();
            break;
        case 'Q':
            if(!Format.empty()) exportHandler(0);
            quit();
            break;
    }
}

void AsyRender::mainLoop()
{
  if(View) {
    GLFWwindow* win = static_cast<GLFWwindow*>(getGLFWWindow());
    glfwRunLoop(win,
      // shouldContinue: continue while window is open
      [win](){ return !glfwWindowShouldClose(win); },

      // shouldDisplay: display when needed
      [this](){ return redraw || redisplay || queueExport; },

      // doDisplay: handle display logic
      [this](){
        redisplay=false;
        waitEvent=true;
        if(resize) {
          fitscreen(!interact::interactive);
          resize=false;
        }
        display();
      },

      // processMessages: dequeue and process messages
      [this](){
        auto const message=messageQueue.dequeue();
        if(message.has_value())
          processMessages(*message);
      },

      // getIdleFunc: return current idle function (or nullptr)
      [this](){ return currentIdleFunc; },

      // shouldWait: use waitEvent to decide between wait and poll
      [this](){ return waitEvent; }
    );
  } else {
    update();
    display();
    if(thread) {
      if(havewindow) {
#ifdef HAVE_PTHREAD
        if(pthread_equal(pthread_self(),this->mainthread))
          exportHandler();
        else
          messageQueue.enqueue(RendererMessage::exportRender);
#endif
      } else {
        initialized=true;
        readyForExport=true;
        exportHandler();
      }
    } else {
      exportHandler();
      quit();
    }
  }
}

} // namespace camp
