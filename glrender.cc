/*****
 * glrender.cc
 * John Bowman and Orest Shardt
 * Render 3D Bezier paths and surfaces.
 *****/

#include <stdlib.h>
#include <fstream>
#include <cstring>
#include <sys/time.h>

#include "picture.h"
#include "common.h"
#include "arcball.h"
#include "bbox3.h"
#include "drawimage.h"
#include "interact.h"
#include "glrender.h"

#ifdef HAVE_LIBGLUT

// For CYGWIN
#ifndef FGAPI
#define FGAPI GLUTAPI
#endif
#ifndef FGAPIENTRY
#define FGAPIENTRY APIENTRY
#endif

#include <GL/freeglut_ext.h>

namespace gl {
  
using camp::picture;
using camp::drawImage;
using camp::transform;
using camp::pair;
using camp::triple;
using vm::array;
using camp::bbox3;
using settings::getSetting;
using settings::Setting;

const double moveFactor=1.0;  
const double zoomFactor=1.05;
const double zoomFactorStep=0.1;
const double spinStep=60.0; // Degrees per second
const double arcballRadius=750.0;
const double resizeStep=1.2;

double Aspect;
bool View;
int Oldpid;
const string* Prefix;
const picture* Picture;
string Format;
int Width,Height;
int oldWidth,oldHeight;
double oWidth,oHeight;

bool Xspin,Yspin,Zspin;
bool Menu;
bool Motion;
int Fitscreen;
int Mode;

double H;
GLint ViewportLimit[2];
double xmin,xmax;
double ymin,ymax;
double zmin,zmax;

double Xmin,Xmax;
double Ymin,Ymax;
double X,Y;

int minimumsize=50; // Minimum initial rendering window width and height

const double degrees=180.0/M_PI;
const double radians=1.0/degrees;

size_t Nlights;
triple *Lights; 
double *Diffuse;
double *Ambient;
double *Specular;
bool ViewportLighting;
bool queueExport=false;

int x0,y0;
int mod;

double lastangle;

double Zoom;
double lastzoom;

float Rotate[16];
float Modelview[16];
Arcball arcball;
  
GLUnurbs *nurb;

int window;
  
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

void lighting(void)
{
  for(size_t i=0; i < Nlights; ++i) {
    GLenum index=GL_LIGHT0+i;
    glEnable(index);
    
    triple Lighti=Lights[i];
    GLfloat position[]={Lighti.getx(),Lighti.gety(),Lighti.getz(),0.0};
    glLightfv(index,GL_POSITION,position);
    
    size_t i4=4*i;
    
    GLfloat diffuse[]={Diffuse[i4],Diffuse[i4+1],Diffuse[i4+2],Diffuse[i4+3]};
    glLightfv(index,GL_DIFFUSE,diffuse);
    
    GLfloat ambient[]={Ambient[i4],Ambient[i4+1],Ambient[i4+2],Ambient[i4+3]};
    glLightfv(index,GL_AMBIENT,ambient);
    
    GLfloat specular[]={Specular[i4],Specular[i4+1],Specular[i4+2],
			Specular[i4+3]};
    glLightfv(index,GL_SPECULAR,specular);
  }
}

void save()
{  
  glFinish();
  glPixelStorei(GL_PACK_ALIGNMENT,1);
  size_t ndata=3*Width*Height;
  unsigned char *data=new unsigned char[ndata];
  if(data) {
    picture pic;
    glReadPixels(0,0,Width,Height,GL_RGB,GL_UNSIGNED_BYTE,data);
    double w=oWidth;
    double h=oHeight;
    double Aspect=((double) Width)/Height;
    if(w > h*Aspect) w=(int) (h*Aspect+0.5);
    else h=(int) (w/Aspect+0.5);
    // Render an antialiased image.
    drawImage *Image=new drawImage(data,Width,Height,
				   transform(0.0,0.0,w,0.0,0.0,h),true);
    pic.append(Image);
    pic.shipout(NULL,*Prefix,Format,0.0,false,View);
    delete Image;
    delete[] data;
   }
}
  
void setProjection()
{
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  double Aspect=((double) Width)/Height;
  double X0=X*(xmax-xmin)/(lastzoom*Width);
  double Y0=Y*(ymax-ymin)/(lastzoom*Height);
  if(H == 0.0) {
    double xsize=Xmax-Xmin;
    double ysize=Ymax-Ymin;
    if(xsize < ysize*Aspect) {
      double r=0.5*ysize*Zoom*Aspect;
      xmin=-r-X0;
      xmax=r-X0;
      ymin=Ymin*Zoom-Y0;
      ymax=Ymax*Zoom-Y0;
    } else {
      double r=0.5*xsize*Zoom/Aspect;
      xmin=Xmin*Zoom-X0;
      xmax=Xmax*Zoom-X0;
      ymin=-r-Y0;
      ymax=r-Y0;
    }
    glOrtho(xmin,xmax,ymin,ymax,-zmax,-zmin);
  } else {
    double r=H*Zoom;
    xmin=-r*Aspect-X0;
    xmax=r*Aspect-X0;
    ymin=-r-Y0;
    ymax=r-Y0;
    glFrustum(xmin,xmax,ymin,ymax,-zmax,-zmin);
  }
  glMatrixMode(GL_MODELVIEW);
  
  arcball.set_params(vec2(0.5*Width,0.5*Height),arcballRadius/Zoom);
}

int screenWidth() 
{
  return min(glutGet(GLUT_SCREEN_WIDTH),ViewportLimit[0]);
}

int screenHeight() 
{
  return min(glutGet(GLUT_SCREEN_HEIGHT),ViewportLimit[1]);
}

bool capsize(int& width, int& height) 
{
  bool resize=false;
  if(width > ViewportLimit[0]) {
    width=ViewportLimit[0];
    resize=true;
  }
  if(height > ViewportLimit[1]) {
    height=ViewportLimit[1];
    resize=true;
  }
  return resize;
}

void reshape0(int width, int height)
{
  X=X/Width*width;
  Y=Y/Height*height;
  
  Width=width;
  Height=height;
  
  setProjection();
  glViewport(0,0,Width,Height);
}
  
void reshape(int width, int height)
{
 if(capsize(width,height))
   glutReshapeWindow(width,height);
 
 reshape0(width,height);
}
  
void windowposition(int& x, int& y, int width=Width, int height=Height)
{
  pair z=getSetting<pair>("position");
  x=(int) z.getx();
  y=(int) z.gety();
  if(x < 0) {
    x += glutGet(GLUT_SCREEN_WIDTH)-width;
    if(x < 0) x=0;
  }
  if(y < 0) {
    y += glutGet(GLUT_SCREEN_HEIGHT)-height;
    if(y < 0) y=0;
  }
}

void setsize(int w, int h, int minsize=0)
{
  int x,y;
  
  if(minsize) {
    if(w < minsize) {
      h=(int) (h*(double) minsize/w+0.5);
      w=minsize;
    }
  
    if(h < minsize) {
      w=(int) (w*(double) minsize/h+0.5);
      h=minsize;
    }
  }
  
  capsize(w,h);
  windowposition(x,y,w,h);
  glutPositionWindow(x,y);
  glutReshapeWindow(w,h);
  reshape0(w,h);
  glutPostRedisplay();
}

void fullscreen() 
{
  int w=screenWidth();
  int h=screenHeight();
  if(w > 0 && h > 0) {
    Width=w;
    Height=h;
#ifdef __CYGWIN__
    glutFullScreen();
#else    
    setsize(w,h,0);
#endif    
  }
}

void Export() 
{
  glReadBuffer(GL_FRONT_LEFT);
  save();
}

void quit() 
{
  glutDestroyWindow(window);
  glutLeaveMainLoop();
}

void disableMenu() 
{
  glutDetachMenu(GLUT_RIGHT_BUTTON);
  Menu=false;
}

void display(void)
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  if(!ViewportLighting) 
    lighting();
    
  triple m(xmin,ymin,zmin);
  triple M(xmax,ymax,zmax);
  double perspective=H == 0.0 ? 0.0 : 1.0/zmax;
  
  bool twosided=settings::getSetting<bool>("twosided");
  
  double size2=hypot(Width,Height);
  
  // Render opaque objects
  Picture->render(nurb,size2,m,M,perspective,false,twosided);
  
  // Enable transparency
  glEnable(GL_BLEND);
  glDepthMask(GL_FALSE);
  
  // Render transparent objects
  Picture->render(nurb,size2,m,M,perspective,true,twosided);
  glDepthMask(GL_TRUE);
  glDisable(GL_BLEND);
  
  if(View) {
    glutSwapBuffers();
    if(queueExport) {
      Export();
      queueExport=false;
    }
    
    int status;
    if(Oldpid != 0 && waitpid(Oldpid, &status, WNOHANG) != Oldpid) {
      kill(Oldpid,SIGHUP);
      Oldpid=0;
    }
  } else {
    glReadBuffer(GL_BACK_LEFT);
    save();
    quit();
  }
}

void update() 
{
  lastzoom=Zoom;
  glLoadIdentity();
  double cz=0.5*(zmin+zmax);
  glTranslatef(0,0,cz);
  glMultMatrixf(Rotate);
  glTranslatef(0,0,-cz);
  glGetFloatv(GL_MODELVIEW_MATRIX,Modelview);
  setProjection();
  glutPostRedisplay();
}

void move(int x, int y)
{
  if(x > 0 && y > 0) {
    X += (x-x0)*Zoom;
    Y += (y0-y)*Zoom;
    x0=x; y0=y;
    update();
  }
}
  
void capzoom() 
{
  static double maxzoom=sqrt(DBL_MAX);
  static double minzoom=1/maxzoom;
  if(Zoom <= minzoom) Zoom=minzoom;
  if(Zoom >= maxzoom) Zoom=maxzoom;
  
}

void zoom(int x, int y)
{
  if(x > 0 && y > 0) {
    if(Menu) {
      disableMenu();
      y0=y;
      return;
    }
    Motion=true;
    static const double limit=log(0.1*DBL_MAX)/log(zoomFactor);
    lastzoom=Zoom;
    double s=zoomFactorStep*(y-y0);
    if(fabs(s) < limit) {
      Zoom *= pow(zoomFactor,s);
      capzoom();
      y0=y;
      setProjection();
      glutPostRedisplay();
    }
  }
}
  
void mousewheel(int wheel, int direction, int x, int y) 
{
  lastzoom=Zoom;
  if(direction > 0)
    Zoom /= zoomFactor;
  else
    Zoom *= zoomFactor;
  
  capzoom();
  setProjection();
  glutPostRedisplay();
}

void rotate(int x, int y)
{
  if(x > 0 && y > 0) {
    if(Menu) {
      disableMenu();
      arcball.mouse_down(x,Height-y);
      return;
    }
    Motion=true;
    arcball.mouse_motion(x,Height-y,0,
			 mod == GLUT_ACTIVE_SHIFT, // X rotation only
			 mod == GLUT_ACTIVE_CTRL);  // Y rotation only

    for(int i=0; i < 4; ++i) {
      const vec4& roti=arcball.rot[i];
      int i4=4*i;
      for(int j=0; j < 4; ++j)
	Rotate[i4+j]=roti[j];
    }
    update();
  }
}
  
double Degrees(int x, int y) 
{
  return atan2(0.5*Height-y-Y,x-0.5*Width-X)*degrees;
}

void updateArcball() 
{
  for(int i=0; i < 4; ++i) {
    int i4=4*i;
    vec4& roti=arcball.rot[i];
    for(int j=0; j < 4; ++j)
      roti[j]=Rotate[i4+j];
  }
  update();
}
  
void rotateX(double step) 
{
  glLoadIdentity();
  glRotatef(step,1,0,0);
  glMultMatrixf(Rotate);
  glGetFloatv(GL_MODELVIEW_MATRIX,Rotate);
  updateArcball();
}

void rotateY(double step) 
{
  glLoadIdentity();
  glRotatef(step,0,1,0);
  glMultMatrixf(Rotate);
  glGetFloatv(GL_MODELVIEW_MATRIX,Rotate);
  updateArcball();
}

void rotateZ(double step) 
{
  glLoadIdentity();
  glRotatef(step,0,0,1);
  glMultMatrixf(Rotate);
  glGetFloatv(GL_MODELVIEW_MATRIX,Rotate);
  updateArcball();
}

void rotateZ(int x, int y)
{
  if(x > 0 && y > 0) {
    if(Menu) {
      disableMenu();
      x=x0; y=y0;
      return;
    }
    Motion=true;
    double angle=Degrees(x,y);
    rotateZ(angle-lastangle);
    lastangle=angle;
  }
}

// Mouse bindings.
// LEFT: rotate
// SHIFT LEFT: zoom
// CTRL LEFT: shift
// MIDDLE: menu
// RIGHT: zoom
// SHIFT RIGHT: rotateX
// CTRL RIGHT: rotateY
// ALT RIGHT: rotateZ
void mouse(int button, int state, int x, int y)
{
  mod=glutGetModifiers();
  
  if(button == GLUT_RIGHT_BUTTON) {
    if(state == GLUT_UP && !Motion) {
      glutAttachMenu(GLUT_RIGHT_BUTTON);
      Menu=true;
      return;
    }
  }
  
  if(Menu) disableMenu();
  else Motion=false;
  
  if(state == GLUT_DOWN) {
    if(button == GLUT_LEFT_BUTTON && mod == GLUT_ACTIVE_CTRL) {
      x0=x; y0=y;
      glutMotionFunc(move);
      return;
    } 
    if((button == GLUT_LEFT_BUTTON && mod == GLUT_ACTIVE_SHIFT) ||
       (button == GLUT_RIGHT_BUTTON && mod == 0)) {
      y0=y;
      glutMotionFunc(zoom);
      return;
    }
    if(button == GLUT_RIGHT_BUTTON && mod == GLUT_ACTIVE_ALT) {
      lastangle=Degrees(x,y);
      glutMotionFunc(rotateZ);
      return;
    }
    arcball.mouse_down(x,Height-y);
    glutMotionFunc(rotate);
  } else
    arcball.mouse_up();
}

timeval lasttime;

double spinstep() 
{
  timeval tv;
  gettimeofday(&tv,NULL);
  double step=spinStep*(tv.tv_sec-lasttime.tv_sec+
			((double) tv.tv_usec-lasttime.tv_usec)/1000000.0);
  lasttime=tv;
  return step;
}

void xspin()
{
  rotateX(spinstep());
}

void yspin()
{
  rotateY(spinstep());
}

void zspin()
{
  rotateZ(spinstep());
}

void initTimer() 
{
  gettimeofday(&lasttime,NULL);
}

void expand() 
{
  setsize((int) (Width*resizeStep+0.5),(int) (Height*resizeStep+0.5));
}

void shrink() 
{
  setsize(max((int) (Width/resizeStep+0.5),1),
	  max((int) (Height/resizeStep+0.5),1));
}

void fitscreen() 
{
  switch(Fitscreen) {
    case 0: // Original size
    {
      setsize(oldWidth,oldHeight,minimumsize);
      ++Fitscreen;
      break;
    }
    case 1: // Fit to screen in one dimension
    {       
      oldWidth=Width;
      oldHeight=Height;
      int w=screenWidth();
      int h=screenHeight();
      if(w > 0 && h > 0) {
	if(w > h*Aspect) w=(int) (h*Aspect+0.5);
	else h=(int) (w/Aspect+0.5);
	setsize(w,h,minimumsize);
      }
      ++Fitscreen;
      break;
    }
    case 2: // Full screen
    {
      fullscreen();
      Fitscreen=0;
      break;
    }
  }
}

void idleFunc(void (*f)())
{
  initTimer();
  glutIdleFunc(f);
}

void mode() 
{
  switch(Mode) {
    case 0:
      for(size_t i=0; i < Nlights; ++i) 
	glEnable(GL_LIGHT0+i);
      glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
      gluNurbsProperty(nurb,GLU_DISPLAY_MODE,GLU_FILL);
      ++Mode;
    break;
    case 1:
      for(size_t i=0; i < Nlights; ++i) 
	glDisable(GL_LIGHT0+i);
      gluNurbsProperty(nurb,GLU_DISPLAY_MODE,GLU_OUTLINE_POLYGON);
      glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
      ++Mode;
    break;
    case 2:
      gluNurbsProperty(nurb,GLU_DISPLAY_MODE,GLU_OUTLINE_PATCH);
      Mode=0;
    break;
  }
  glutPostRedisplay();
}

void idle() 
{
  glutIdleFunc(NULL);
  Xspin=Yspin=Zspin=false;
}

void spinx() 
{
  if(Xspin)
    idle();
  else {
    idleFunc(xspin);
    Xspin=true;
    Yspin=Zspin=false;
  }
}

void spiny()
{
  if(Yspin)
    idle();
  else {
    idleFunc(yspin);
    Yspin=true;
    Xspin=Zspin=false;
  }
}

void spinz()
{
  if(Zspin)
    idle();
  else {
    idleFunc(zspin);
    Zspin=true;
    Xspin=Yspin=false;
  }
}

void home() 
{
  idle();
  X=Y=0.0;
  arcball.init();
  glLoadIdentity();
  glGetFloatv(GL_MODELVIEW_MATRIX,Rotate);
  glGetFloatv(GL_MODELVIEW_MATRIX,Modelview);
  lastzoom=Zoom=1.0;
}

void keyboard(unsigned char key, int x, int y)
{
  switch(key) {
    case 'h':
      home();
      update();
      break;
    case 'f':
      fitscreen();
      break;
    case 'x':
      spinx();
      break;
    case 'y':
      spiny();
      break;
    case 'z':
      spinz();
      break;
    case 's':
      idle();
      break;
    case 'm':
      mode();
      break;
    case 'e':
      Export();
      break;
    case '+':
    case '=':
      expand();
      break;
    case '-':
    case '_':
      shrink();
      break;
    case 17: // Ctrl-q
    case 'q':
      if(!Format.empty()) Export();
      quit();
      break;
  }
}
 
enum Menu {HOME,FITSCREEN,XSPIN,YSPIN,ZSPIN,STOP,MODE,EXPORT,QUIT};

void menu(int choice)
{
  disableMenu();
  Motion=true;
  switch (choice) {
    case HOME: // Home
      home();
      update();
      break;
    case FITSCREEN:
      fitscreen();
      break;
    case XSPIN:
      spinx();
      break;
    case YSPIN:
      spiny();
      break;
    case ZSPIN:
      spinz();
      break;
    case STOP:
      idle();
      break;
    case MODE:
      mode();
      break;
    case EXPORT:
      queueExport=true;
      break;
    case QUIT:
      quit();
      break;
  }
}

// angle=0 means orthographic.
void glrender(const string& prefix, const picture *pic, const string& format,
	      double width, double height,
	      double angle, const triple& m, const triple& M,
	      size_t nlights, triple *lights, double *diffuse,
	      double *ambient, double *specular, bool Viewportlighting,
	      bool view, int oldpid)
{
  if(width <= 0 || height <= 0) return;
  
  Prefix=&prefix;
  Picture=pic;
  Format=format;
  View=view;
  Oldpid=oldpid;
  Nlights=min(nlights,(size_t) GL_MAX_LIGHTS);
  Lights=lights;
  Diffuse=diffuse;
  Ambient=ambient;
  Specular=specular;
  ViewportLighting=Viewportlighting;
    
  Xmin=m.getx();
  Xmax=M.getx();
  Ymin=m.gety();
  Ymax=M.gety();
  zmin=m.getz();
  zmax=M.getz();
  H=angle != 0.0 ? -tan(0.5*angle*radians)*zmax : 0.0;
   
  Menu=false;
  Motion=true;
  Fitscreen=0;
  Mode=0;
  
  string options=string(settings::argv0)+" ";
#ifndef __CYGWIN__
  if(!View) options += "-iconic ";
#endif  
  options += getSetting<string>("glOptions");
  char **argv=args(options.c_str(),true);
  int argc=0;
  while(argv[argc] != NULL)
    ++argc;
  
  bool screen=View && Format.empty();
  
  double expand=getSetting<double>("render");
  if(expand < 0)
    expand *= screen ? -1.0 : 
      (Format.empty() || Format == "eps" || Format == "pdf" ? -4.0 : -2.0);
  
  glutInit(&argc,argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
  
  glutInitWindowSize(1,1);
  window=glutCreateWindow("");
  glGetIntegerv(GL_MAX_VIEWPORT_DIMS, ViewportLimit);
  glutDestroyWindow(window);

  // Force a hard viewport limit to work around direct rendering bugs.
  // Alternatively, one can use -glOptions=-indirect (with a performance
  // penalty).
  int limit=(int) getSetting<Int>("maxviewport");
  if(limit > 0) {
    ViewportLimit[0]=min(ViewportLimit[0],limit);
    ViewportLimit[1]=min(ViewportLimit[1],limit);
  }

#ifdef __CYGWIN__
  int margin=60;
  ViewportLimit[0]=min(ViewportLimit[0],screenWidth()-margin);
  ViewportLimit[1]=min(ViewportLimit[1],screenHeight()-margin);
#endif
  
  oWidth=width;
  oHeight=height;
  Aspect=((double) width)/height;
  
  Width=min((int) (ceil(expand*width)),ViewportLimit[0]);
  Height=min((int) (ceil(expand*height)),ViewportLimit[1]);
  
  if(Width > Height*Aspect) 
    Width=min((int) (ceil(Height*Aspect)),ViewportLimit[0]);
  else 
    Height=min((int) (ceil(Width/Aspect)),ViewportLimit[1]);
  
  Aspect=((double) Width)/Height;
  
  if(settings::verbose > 1) 
    cout << "Rendering " << prefix << " as " << Width << "x" << Height
	 << " image" << endl;
    
  glMatrixMode(GL_MODELVIEW);
  home();
  
  int x,y;
  windowposition(x,y);
  glutInitWindowPosition(x,y);
  
  glutInitWindowSize(Width,Height);
  window=glutCreateWindow(((prefix == "out" ? "Asymptote" : prefix)+
			   " [Double click right button for menu]").c_str());
  oldWidth=Width;
  oldHeight=Height;
  
  glClearColor(1.0,1.0,1.0,0.0);
   
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_MAP1_VERTEX_3);
  glEnable(GL_MAP2_VERTEX_3);
  
  glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
  
  nurb=gluNewNurbsRenderer();
  gluNurbsProperty(nurb,GLU_SAMPLING_METHOD,GLU_PARAMETRIC_ERROR);
  gluNurbsProperty(nurb,GLU_SAMPLING_TOLERANCE,0.5);
  gluNurbsProperty(nurb,GLU_PARAMETRIC_TOLERANCE,1.0);
  gluNurbsProperty(nurb,GLU_CULLING,GLU_TRUE);
  
  // The callback tesselation algorithm avoids artifacts at degenerate control
  // points.
  gluNurbsProperty(nurb,GLU_NURBS_MODE,GLU_NURBS_TESSELLATOR);
  gluNurbsCallback(nurb,GLU_NURBS_BEGIN,(_GLUfuncptr) glBegin);
  gluNurbsCallback(nurb,GLU_NURBS_VERTEX,(_GLUfuncptr) glVertex3fv);
  gluNurbsCallback(nurb,GLU_NURBS_END,(_GLUfuncptr) glEnd);
  mode();
  
  glEnable(GL_LIGHTING);
  if(getSetting<bool>("twosided"))
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE,GL_TRUE);
  
  if(ViewportLighting)
    lighting();
  
  glutReshapeFunc(reshape);
  glutDisplayFunc(display);
  glutKeyboardFunc(keyboard);
  glutMouseFunc(mouse);
  glutMouseWheelFunc(mousewheel);
   
  glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,GLUT_ACTION_CONTINUE_EXECUTION);

  glutCreateMenu(menu);
  glutAddMenuEntry("(h) Home",HOME);
  glutAddMenuEntry("(f) Fitscreen",FITSCREEN);
  glutAddMenuEntry("(x) X spin",XSPIN);
  glutAddMenuEntry("(y) Y spin",YSPIN);
  glutAddMenuEntry("(z) Z spin",ZSPIN);
  glutAddMenuEntry("(s) Stop",STOP);
  glutAddMenuEntry("(m) Mode",MODE);
  glutAddMenuEntry("(e) Export",EXPORT);
  glutAddMenuEntry("(q) Quit" ,QUIT);
  
  glutAttachMenu(GLUT_MIDDLE_BUTTON);

  if(screen && !interact::interactive && getSetting<bool>("fitscreen"))
    Fitscreen=1;
  
  fitscreen();
  
  glutMainLoop();
}
  
} // namespace gl

#endif
