#include "renderBase.h"
#include "glfw.h"
#include "settings.h"
#include "drawelement.h"
#include "interact.h"
#include "picture.h"

#ifdef HAVE_LIBGLM

#ifdef HAVE_GLFW
// Forward declaration for GLFWwindow to avoid including glfw3.h here
struct GLFWwindow;
#endif

using settings::getSetting;


namespace camp {

AsyRender* gl;

} // namespace camp

using namespace glm;

// Forward declaration for exit handler (defined in exithandlers.h)
void exitHandler(int);

namespace camp
{

void AsyRender::copyRenderArgs(RenderFunctionArgs const& args)
{
  // Basic picture and format state
  pic = args.pic;
  Prefix = args.prefix;
  Format = args.format;
  remesh = true;

  // Lighting
  nlights = args.nlightsin;
  Lights = args.lights;
  LightsDiffuse = args.diffuse;
  Oldpid = args.oldpid;

  // Camera parameters
  Angle = args.angle * radians;
  lastzoom = 0;
  Zoom0 = std::fpclassify(args.zoom) == FP_NORMAL ? args.zoom : 1.0;
  Shift = args.shift / Zoom0;
  Margin = args.margin;

  // Background color
  for (int i = 0; i < 4; i++)
    Background[i] = static_cast<float>(args.background[i]);

  // View settings
  ViewExport = args.view;
  View = args.view && !settings::getSetting<bool>("offscreen");

  title = std::string(PACKAGE_NAME) + ": " + args.prefix.c_str();

  // Tile size limits from -maxtile setting
  {
    pair maxtile = getSetting<pair>("maxtile");
    maxTileWidth = (int)maxtile.getx();
    maxTileHeight = (int)maxtile.gety();
    if (maxTileWidth <= 0) maxTileWidth = 1024;
    if (maxTileHeight <= 0) maxTileHeight = 768;
  }

  // Scene bounds
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

  // Transform matrices
  for (int i = 0; i < 16; ++i)
    T[i] = args.t[i];

  for (int i = 0; i < 16; ++i)
    Tup[i] = args.tup[i];
}

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
  // Guard against zero dimensions to prevent division by zero (SIGFPE).
  if(Width <= 0) Width = 1;
  if(Height <= 0) Height = 1;

  double aspect = ((double) Width) / Height;
  double zoom = Zoom * zoomFactor;
  double xshift = (X / (double) Width + Shift.getx() * Xfactor) * zoom;
  double yshift = (Y / (double) Height + Shift.gety() * Yfactor) * zoom;
  double zoominv = 1.0 / zoom;
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
#ifdef HAVE_LIBGLFW
  if(View) {
    pthread_t postThread;
    if(pthread_create(&postThread,NULL,postEmptyEvent,NULL) == 0)
      pthread_join(postThread,NULL);
  }
#endif
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


void AsyRender::prepareScene()
{

#ifdef HAVE_PTHREAD
  static bool first=true;
  if(threads && first) {
    threadMgr.wait(threadMgr.initSignal,threadMgr.initLock);
    threadMgr.endwait(threadMgr.initSignal,threadMgr.initLock);
    first=false;
  }

  if(format3dWait)
    threadMgr.wait(threadMgr.initSignal,threadMgr.initLock);
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

    setOpaque();
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

void AsyRender::fitAspect(int& w, int& h)
{
  if(w > h * Aspect)
    w = (int) std::ceil(h * Aspect);
  else
    h = (int) std::ceil(w / Aspect);
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

/**
 * Set window to fullscreen size.
 * Base implementation handles dimension calculation and GLFW window operations.
 */
void AsyRender::fullscreen(bool reposition)
{
  Xfactor = Yfactor = 1.0;
  if (screenWidth < screenHeight * Aspect)
    zoomFactor = (double)screenWidth / (screenHeight * Aspect);
  else
    zoomFactor = 1.0;
  setsize(screenWidth, screenHeight, reposition);
  reshape(screenWidth, screenHeight);
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
  remesh = true;
  switch(Fitscreen) {
    case 0: // Original size: use saved framebuffer dimensions
    {
      Xfactor = Yfactor = 1.0;
      zoomFactor = 1.0;
      setsize(oldWidth, oldHeight, reposition);
      break;
    }
    case 1: // Fit to screen: screenWidth/screenHeight already physical pixels
    {
      zoomFactor = 1.0;
      int w = screenWidth;
      int h = screenHeight;
      fitAspect(w, h);
      setsize(w, h, reposition);
      reshape(w,h);
      break;
    }
    case 2: // Full screen: fill physical screen directly
    {
      fullscreen(reposition);
      break;
    }
  }
}

void AsyRender::toggleFitScreen()
{
  Fitscreen = (Fitscreen + 1) % 3;
  fitscreen();
}

void AsyRender::home()
{
  idle();
  X = Y = cx = cy = 0;
  rotateMat = viewMat = dmat4(1.0);
  lastzoom = Zoom = Zoom0;
  framecount = 0;

  remesh = true;
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
  if(!threads) {
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

#ifdef HAVE_LIBGLFW
/**
 * Show the window if hidden (GLFW-specific implementation).
 */
void AsyRender::showWindow()
{
  GLFWwindow* win = glfwWindow;
  if(View && !hideWindow && !glfwGetWindowAttrib(win, GLFW_VISIBLE))
    ::glfwShowWindow(win);
}
#else
// Stub for when GLFW is unavailable (satisfies vtable)
void AsyRender::showWindow() {}
#endif

/**
 * Window close handler (library-agnostic).
 */
void AsyRender::onClose()
{
  // Default: no-op - derived classes can override for specific behavior
}

#ifdef HAVE_LIBGLFW
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
            redraw = true;
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

void AsyRender::onScroll(double xoffset, double yoffset)
{
    std::string action = getGLFWScrollAction(yoffset <= 0);

    auto zoomFactor = getSetting<double>("zoomfactor");
    if(action == "zoomin" || action.empty()) {
        if(zoomFactor > 0.0) Zoom /= zoomFactor;
    } else if(action == "zoomout") {
        if(zoomFactor > 0.0) Zoom *= zoomFactor;
    }
    update();
}

#else // !HAVE_LIBGLFW
// Stubs for when GLFW is unavailable (satisfy vtable)
void AsyRender::onKey(int, int, int, int) {}
void AsyRender::onScroll(double, double) {}
void AsyRender::onMouseButton(int, int, int) {}
#endif // HAVE_LIBGLFW

void AsyRender::onCursorPos(double xpos, double ypos)
{
    if (lastAction == "rotate") {
        Arcball arcball(xprev * 2 / Width - 1, 1 - yprev * 2 / Height,
                        xpos * 2 / Width - 1, 1 - ypos * 2 / Height);
        triple axis = arcball.axis;
        rotateMat = rotate(2 * arcball.angle / Zoom * ArcballFactor,
                           dvec3(axis.getx(), axis.gety(), axis.getz())) * rotateMat;
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

void AsyRender::onFramebufferResize(int width, int height)
{
  if(width == 0 || height == 0) return;
  if(width == Width && height == Height) return;
  reshape(width, height);
  update();
  remesh = true;
}

#ifdef HAVE_LIBGLFW
void AsyRender::mainLoop()
{
  if(View) {
    GLFWwindow* win = glfwWindow;
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
#ifdef HAVE_PTHREAD
      [this](){
        auto const message=threadMgr.messageQueue.dequeue();
        if(message.has_value())
          processMessages(*message);
      },
#else
      [](){},
#endif

      // getIdleFunc: return current idle function (or nullptr)
      [this](){ return currentIdleFunc; },

      // shouldWait: use waitEvent to decide between wait and poll
      [this](){ return waitEvent; }
    );
  } else {
    update();
    display();
    if(threads) {
      if(havewindow) {
#ifdef HAVE_PTHREAD
        if(pthread_equal(pthread_self(),threadMgr.mainthread))
          exportHandler(0);
        else
          threadMgr.messageQueue.enqueue(RendererMessage::exportRender);
#endif
      } else {
        initialized=true;
        readyForExport=true;
        exportHandler(0);
#ifdef HAVE_PTHREAD
        // Notify main thread only when called from the render thread.
        // When called directly from the main thread, there's nobody to wake.
        if(!pthread_equal(pthread_self(), threadMgr.mainthread))
          threadMgr.endwait(threadMgr.readySignal, threadMgr.readyLock);
#endif
      }
    } else {
      exportHandler(0);
      quit();
    }
  }
}
#else // !HAVE_LIBGLFW
// Stub for when GLFW is unavailable (satisfies vtable)
void AsyRender::mainLoop() {}
#endif // HAVE_LIBGLFW

#ifdef HAVE_RENDERER

// =========================================================================
// Consolidated renderer-specific function definitions.
// All functions below require HAVE_RENDERER (implies HAVE_LIBGLM + HAVE_LIBGLFW).
// They are called only from vkrender.cc and glrender.cc.
// =========================================================================

// Matrix accessor functions - shared between GL and Vulkan renderers.
// These delegate to the corresponding AsyRender member functions.
const glm::dmat4& getProjViewMat() { return gl->getProjViewMat(); }
const glm::dmat4& getViewMat()     { return gl->getViewMat(); }
const glm::dmat3& getNormMat()     { return gl->getNormMat(); }

void AsyRender::initDisplay(int contentWidth, int contentHeight)
{
  // Compute expand/fullWidth/fullHeight (unscaled content dimensions).
  double expand = settings::getSetting<double>("render");
  if (expand < 0)
    expand *= (Format.empty() || Format == "eps" || Format == "pdf") ? -2.0 : -1.0;
  if (antialias) expand *= 2.0;

  fullWidth = (int) std::ceil(expand * contentWidth);
  fullHeight = (int) std::ceil(expand * contentHeight);

  // Guard against zero/negative dimensions from empty/degenerate scenes.
  if(fullWidth <= 0) fullWidth = 1;
  if(fullHeight <= 0) fullHeight = 1;

  oWidth = contentWidth;
  oHeight = contentHeight;

  GLFWmonitor* monitor = NULL;
  glfwInit();

  devicePixelRatio = settings::getSetting<double>("devicepixelratio");
  monitor = glfwGetPrimaryMonitor();
  if (monitor) {
    int mx, my;
    glfwGetMonitorWorkarea(monitor, &mx, &my, &screenWidth, &screenHeight);
    if (devicePixelRatio <= 0.0) {
      float sx = 1.0f, sy = 1.0f;
      glfwGetMonitorContentScale(monitor, &sx, &sy);
      devicePixelRatio = std::max(sx, sy);
    }
  } else {
    screenWidth = fullWidth;
    screenHeight = fullHeight;
  }

  oldWidth = (int) std::ceil(contentWidth * devicePixelRatio);
  oldHeight = (int) std::ceil(contentHeight * devicePixelRatio);

  int w = std::min(oldWidth, screenWidth);
  int h = std::min(oldHeight, screenHeight);

  fitAspect(w, h);

  if(View) {
    Width = w;
    Height = h;
  } else {
    // For offscreen rendering, use a framebuffer large enough for efficient
    // tiling. OpenGL and Vulkan both tile in Export() to produce the final
    // fullWidth x fullHeight image, so the GPU framebuffer does not need to
    // be allocated at the expanded resolution. However, it should be at least
    // as large as the desired max tile size (1024x768) to avoid wasting GPU
    // bandwidth on excessive tile overhead.  Cap at the full export resolution
    // since larger tiles provide no benefit.
    int minTileW = 1024;
    int minTileH = 768;
    Width  = std::max(w, std::min(minTileW, fullWidth));
    Height = std::max(h, std::min(minTileH, fullHeight));
    // Ensure aspect ratio is preserved.
    if ((double)Width / Height > (double)fullWidth / fullHeight)
      Width = (int)std::ceil(Height * (double)fullWidth / fullHeight);
    else
      Height = (int)std::ceil(Width * (double)fullHeight / fullWidth);
  }

  // Guard against zero dimensions (e.g., headless rendering with no monitor)
  // to avoid division by zero in setDimensions() and ArcballFactor computation.
  if(Width <= 0) Width = 1;
  if(Height <= 0) Height = 1;

  home();

  ArcballFactor = 1 + 8.0 * hypot(Margin.getx(), Margin.gety()) / hypot(Width, Height);
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

void AsyRender::setOpaque()
{
  Opaque = transparentData.indices.empty();
}

void AsyRender::exportHandler(int)
{
  readyAfterExport = true;
}

void AsyRender::setsize(int w, int h, bool reposition)
{
  capsize(w, h);
  if (View && glfwWindow != nullptr) {
    GLFWwindow* win = glfwWindow;
    ::glfwSetWindowSize(win, w, h);
    if (reposition) {
      int x, y;
      windowposition(x, y, w, h);
      ::glfwSetWindowPos(win, x, y);
    }
  }
  update();
}

void AsyRender::updateHandler(int)
{
  if (View && !interact::interactive) {
    ::glfwHideWindow(glfwWindow);
    if (!getSetting<bool>("fitscreen"))
      Fitscreen = 0;
  }
  resize = true;
  redisplay = true;
  redraw = true;
  remesh = true;
  waitEvent = false;
}

void AsyRender::quit()
{
  resize = false;
  waitEvent = false;
  redraw = false;

  if (threads) {
#ifdef HAVE_PTHREAD
    if (!interact::interactive) {
      idle();
      threadMgr.endwait(threadMgr.readySignal, threadMgr.readyLock);
    }
#endif
    if (View && glfwWindow) {
      ::glfwHideWindow(glfwWindow);
      hideWindow = true;
    }
  } else {
    finalizeProcess();
    if (View && glfwWindow) {
      ::glfwDestroyWindow(glfwWindow);
      glfwWindow = nullptr;
    }
    glfwTerminate();
    exit(0);
  }
}

void AsyRender::onMouseButton(int button, int action, int mods)
{
    auto const currentActionStr = getGLFWAction(button, mods);
    if (currentActionStr.empty()) return;
    if (action == GLFW_PRESS) {
        lastAction = currentActionStr;
        double xpos, ypos;
        glfwGetCursorPos(glfwWindow, &xpos, &ypos);
        xprev = xpos;
        yprev = ypos;
    } else if (action == GLFW_RELEASE) {
        lastAction.clear();
    }
}

#else // !HAVE_RENDERER
// Stubs for when GLFW/Vulkan/GL are unavailable (satisfy vtable and link).
void AsyRender::clearData() {}
void AsyRender::setOpaque() {}
void AsyRender::exportHandler(int) { readyAfterExport = true; }
void AsyRender::setsize(int, int, bool) {}
void AsyRender::updateHandler(int) {}
void AsyRender::quit() { exit(0); }

#endif // HAVE_RENDERER

} // namespace camp

#endif // HAVE_LIBGLM
