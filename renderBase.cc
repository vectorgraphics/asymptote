#include "renderBase.h"
#include "settings.h"
#include "drawelement.h"

using settings::getSetting;
using namespace glm;

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
  // Default implementation - derived classes should override
}

void AsyRender::setProjection()
{
  // Default implementation - derived classes should override
}

void AsyRender::updateModelViewData()
{
  // Default implementation - derived classes should override
}

void AsyRender::update()
{
  // Default implementation - derived classes should override
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
  mode = DrawMode((mode + 1) % DRAWMODE_MAX);
  remesh = true;
  redraw = true;
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
  // Default implementation - derived classes should override
}

void AsyRender::shrink()
{
  // Default implementation - derived classes should override
}

void AsyRender::exportHandler(int)
{
  // Default implementation - derived classes should override
}

void AsyRender::quit()
{
  // Default implementation - derived classes should override
}

} // namespace camp
