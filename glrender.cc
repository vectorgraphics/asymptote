/*****
 * glrender.cc
 * John Bowman and Orest Shardt
 * Render 3D Bezier paths and surfaces.
 *****/

/*
 * Copyright (c) 1993-1997, Silicon Graphics, Inc.
 * ALL RIGHTS RESERVED 
 * Permission to use, copy, modify, and distribute this software for 
 * any purpose and without fee is hereby granted, provided that the above
 * copyright notice appear in all copies and that both the copyright notice
 * and this permission notice appear in supporting documentation, and that 
 * the name of Silicon Graphics, Inc. not be used in advertising
 * or publicity pertaining to distribution of the software without specific,
 * written prior permission. 
 *
 * THE MATERIAL EMBODIED ON THIS SOFTWARE IS PROVIDED TO YOU "AS-IS"
 * AND WITHOUT WARRANTY OF ANY KIND, EXPRESS, IMPLIED OR OTHERWISE,
 * INCLUDING WITHOUT LIMITATION, ANY WARRANTY OF MERCHANTABILITY OR
 * FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL SILICON
 * GRAPHICS, INC.  BE LIABLE TO YOU OR ANYONE ELSE FOR ANY DIRECT,
 * SPECIAL, INCIDENTAL, INDIRECT OR CONSEQUENTIAL DAMAGES OF ANY
 * KIND, OR ANY DAMAGES WHATSOEVER, INCLUDING WITHOUT LIMITATION,
 * LOSS OF PROFIT, LOSS OF USE, SAVINGS OR REVENUE, OR THE CLAIMS OF
 * THIRD PARTIES, WHETHER OR NOT SILICON GRAPHICS, INC.  HAS BEEN
 * ADVISED OF THE POSSIBILITY OF SUCH LOSS, HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, ARISING OUT OF OR IN CONNECTION WITH THE
 * POSSESSION, USE OR PERFORMANCE OF THIS SOFTWARE.
 * 
 * US Government Users Restricted Rights 
 * Use, duplication, or disclosure by the Government is subject to
 * restrictions set forth in FAR 52.227.19(c)(2) or subparagraph
 * (c)(1)(ii) of the Rights in Technical Data and Computer Software
 * clause at DFARS 252.227-7013 and/or in similar or successor
 * clauses in the FAR or the DOD or NASA FAR Supplement.
 * Unpublished-- rights reserved under the copyright laws of the
 * United States.  Contractor/manufacturer is Silicon Graphics,
 * Inc., 2011 N.  Shoreline Blvd., Mountain View, CA 94039-7311.
 *
 * OpenGL(R) is a registered trademark of Silicon Graphics, Inc.
 */

#include <stdlib.h>
#include <fstream>
#include "picture.h"
#include "common.h"
#include "arcball.h"
#include "bbox3.h"
#include "drawimage.h"

#ifdef HAVE_LIBGLUT
#include <GL/glut.h>
#include <GL/freeglut_ext.h>

namespace gl {
  
using camp::picture;
using camp::drawImage;
using camp::transform;
using camp::triple;
using vm::array;
using camp::bbox3;
using settings::getSetting;

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

bool View;
bool Save;
int Oldpid;
const char* Prefix;
picture* Picture;
string Format;
int Width=0;
int Height=0;

double H;
unsigned char **Data;
GLint viewportLimit[2];
triple Light; 
double xmin,xmax;
double ymin,ymax;
double zmin,zmax;

double cx;
double Ymin,Ymax;
double X,Y;

const double moveFactor=1.0;  
const double zoomFactor=1.05;
const double zoomFactorStep=0.25;
const double spinStep=5.0;
const float arcballRadius=500.0;

const double degrees=180.0/M_PI;
const double radians=1.0/degrees;

int x0,y0;
int mod;

bool spinning;

double xangle;
double yangle;
double zangle;
double lastangle;

double Zoom=1.0;
double lastzoom=1.0;

float Rotate[16];
float Modelview[16];

int window;
  
void initlights(void)
{
  GLfloat ambient[]={0.1, 0.1, 0.1, 1.0};
  GLfloat position[]={Light.getx(), Light.gety(), Light.getz(), 0.0};

  if(getSetting<bool>("twosided")) {
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE,GL_TRUE);
    // GL_LIGHT_MODEL_TWO_SIDE seems to require CW orientation.  
    glFrontFace(GL_CW);
  }
  
  glEnable(GL_LIGHTING);
  glEnable(GL_LIGHT0);
  glLightfv(GL_LIGHT0,GL_AMBIENT,ambient);
  glLightfv(GL_LIGHT0,GL_POSITION,position);
}

void save()
{  
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  size_t ndata=3*Width*Height; // Use ColorComponents[colorspace]
  unsigned char data[ndata];
  glReadPixels(0,0,Width,Height,GL_RGB,GL_UNSIGNED_BYTE,data);
  Int expand=getSetting<Int>("render");
  if(expand <= 0) expand=1;
  double f=1.0/expand;
  Picture->append(new drawImage(data,Width,Height,
				transform(0.0,0.0,Width*f,0.0,0.0,Height*f)));
  Picture->shipout(NULL,Prefix,Format);
}
  
void quit() 
{
  glutDestroyWindow(window);
  glutLeaveMainLoop();
}

triple transform(double x, double y, double z) 
{
  x -= Modelview[12];
  y -= Modelview[13];
  z -= Modelview[14];

  return triple((Modelview[0]*x+Modelview[1]*y+Modelview[2]*z),
		(Modelview[4]*x+Modelview[5]*y+Modelview[6]*z),
		(Modelview[8]*x+Modelview[9]*y+Modelview[10]*z));
}

void display(void)
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  bbox3 b(transform(xmin,ymin,zmax));
  b.addnonempty(transform(xmin,ymax,zmax));
  b.addnonempty(transform(xmax,ymin,zmax));
  b.addnonempty(transform(xmax,ymax,zmax));
  if(H == 0) {
    b.addnonempty(transform(xmin,ymin,zmin));
    b.addnonempty(transform(xmin,ymax,zmin));
    b.addnonempty(transform(xmax,ymin,zmin));
    b.addnonempty(transform(xmax,ymax,zmin));
  } else {
    double f=zmin/zmax;
    b.addnonempty(transform(xmin*f,ymin*f,zmin));
    b.addnonempty(transform(xmin*f,ymax*f,zmin));
    b.addnonempty(transform(xmax*f,ymin*f,zmin));
    b.addnonempty(transform(xmax*f,ymax*f,zmin));
  }
  
  // Render opaque objects
  Picture->render(Width,Height,Zoom,b,false);
  
  // Enable transparency
  glEnable(GL_BLEND);
  glDepthMask(GL_FALSE);
  glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
  
  // Render transparent objects
  Picture->render(Width,Height,Zoom,b,true);
  glDepthMask(GL_TRUE);
  glDisable(GL_BLEND);
  
  if(View) {
    glutSwapBuffers();
    int status;
    if(Oldpid != 0 && waitpid(Oldpid, &status, WNOHANG) != Oldpid) {
      kill(Oldpid,SIGHUP);
      Oldpid=0;
    }
  } else {
    glReadBuffer(GL_BACK_LEFT);
    if(Save) save();
    quit();
  }
}

Arcball arcball;
  
void setProjection()
{
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  double Aspect=((double) Width)/Height;
  double X0=X*(xmax-xmin)/(lastzoom*Width);
  double Y0=Y*(ymax-ymin)/(lastzoom*Height);
  if(H == 0.0) {
    double factor=0.5*(Ymax-Ymin)*Zoom*Aspect;
    X0 -= cx;
    xmin=-factor-X0;
    xmax=factor-X0;
    ymin=Ymin*Zoom-Y0;
    ymax=Ymax*Zoom-Y0;
    glOrtho(xmin,xmax,ymin,ymax,-zmax,-zmin);
  } else {
    double r=H*Zoom;
    xmin=-r*Aspect-X0;
    xmax=r*Aspect-X0;
    ymin=-r-Y0;
    ymax=r-Y0;
    glFrustum(xmin,xmax,ymin,ymax,-zmax,-zmin);
  }
  arcball.set_params(vec2(0.5*Width,0.5*Height),arcballRadius/Zoom);
}

void reshape(int width, int height)
{
  bool Reshape=false;
  if(width > viewportLimit[0]) {
    width=viewportLimit[0];
    Reshape=true;
  }
  if(height > viewportLimit[1]) {
    height=viewportLimit[1];
    Reshape=true;
  }
  if(Reshape)
    glutReshapeWindow(width,height);
  
  X=X/Width*width;
  Y=Y/Height*height;
  
  Width=width;
  Height=height;
  
  setProjection();
  glViewport(0,0,Width,Height);
  
}
  
void keyboard(unsigned char key, int x, int y)
{
  switch(key) {
    case 'x':
    {
      glReadBuffer(GL_FRONT_LEFT);
      save();
      break;
    }
    case 17: // Ctrl-q
    case 'q':
    {
      glReadBuffer(GL_FRONT_LEFT);
      if(Save) save();
      quit();
      break;
    }
  }
}
 
void update() 
{
  glMatrixMode(GL_MODELVIEW);
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
  lastzoom=Zoom;
  if(x > 0 && y > 0) {
    X += (x-x0)*Zoom;
    Y += (y0-y)*Zoom;
    x0=x; y0=y;
    update();
  }
}
  
void zoom(int x, int y)
{
  static const double limit=log(0.1*DBL_MAX)/log(zoomFactor);
  if(x > 0 && y > 0) {
    lastzoom=Zoom;
    double s=zoomFactorStep*(y-y0);
    if(fabs(s) < limit) {
      Zoom *= pow(zoomFactor,s);
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
  
  setProjection();
  glutPostRedisplay();
}

void rotate(int x, int y)
{
  if(spinning) return;
  arcball.mouse_motion(x,Height-y,0,
		       mod == GLUT_ACTIVE_SHIFT, // X rotation only
		       mod == GLUT_ACTIVE_CTRL);  // Y rotation only

  for(int i=0; i < 4; ++i) {
    vec4 roti=arcball.rot[i];
    int i4=4*i;
    for(int j=0; j < 4; ++j)
      Rotate[i4+j]=roti[j];
  }
  update();
}
  
double Degrees(int x, int y) 
{
  return atan2(0.5*Height-y-Y,x-0.5*Width-X)*degrees;
}

void rotateX(double step) 
{
  xangle += step;
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glRotatef(xangle,1,0,0);
  glGetFloatv(GL_MODELVIEW_MATRIX,Rotate);
  update();
}

void rotateY(double step) 
{
  yangle += step;
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glRotatef(yangle,0,1,0);
  glGetFloatv(GL_MODELVIEW_MATRIX,Rotate);
  update();
}

void rotateZ(double step) 
{
  zangle += step;
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glRotatef(zangle,0,0,1);
  glGetFloatv(GL_MODELVIEW_MATRIX,Rotate);
  update();
}

void rotateZ(int x, int y)
{
  double angle=Degrees(x,y);
  rotateZ(angle-lastangle);
  lastangle=angle;
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

void Xspin()
{
  rotateX(spinStep);
}

void Yspin()
{
  rotateY(spinStep);
}

void Zspin()
{
  rotateZ(spinStep);
}

void stopSpinning() 
{
  glutIdleFunc(NULL);
  spinning=false;
}

void menu(int choice)
{
  switch (choice) {
    case 1: // Home
      stopSpinning();
      X=Y=0.0;
      arcball.init();
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();
      glGetFloatv(GL_MODELVIEW_MATRIX,Rotate);
      xangle=yangle=zangle=0.0;
      update();
      break;
    case 2: // X spin
      spinning=true;
      glutIdleFunc(Xspin);
      break;
    case 3: // Y spin
      spinning=true;
      glutIdleFunc(Yspin);
      break;
    case 4: // Z spin
      spinning=true;
      glutIdleFunc(Zspin);
      break;
    case 5: // Stop
      stopSpinning();
      // Update arcball
      for(int i=0; i < 4; ++i) {
	int i4=4*i;
	for(int j=0; j < 4; ++j)
	  arcball.rot[i][j]=Rotate[i4+j];
      }
      break;
    case 6: // Export
      glReadBuffer(GL_FRONT_LEFT);
      save();
      break;
    case 7: // Quit
      quit();
      break;
  }
}

// angle=0 means orthographic.
void glrender(const string& prefix, picture *pic, const string& format,
	      int& width, int& height, const triple& light,
	      double angle, const triple& m, const triple& M, bool view,
	      int oldpid)
{
  Prefix=prefix.c_str();
  Picture=pic;
  Format=format;
  View=view;
  Save=!view || !format.empty();
  Oldpid=oldpid;
  Light=light;
  cx=0.5*(m.getx()+M.getx());
  Ymin=m.gety();
  Ymax=M.gety();
  zmin=m.getz();
  zmax=M.getz();
  H=angle != 0.0 ? -tan(0.5*angle*radians)*zmax : 0.0;
   
  X=Y=0.0;
  xangle=yangle=zangle=0.0;
  spinning=false;
  
  string options=string(settings::argv0)+" ";
  if(!View) options += "-iconic ";
  options += settings::getSetting<string>("glOptions");
  char **argv=args(options.c_str(),true);
  int argc=0;
  while(argv[argc] != NULL)
    ++argc;
  glutInit(&argc,argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
  glutInitWindowPosition(0,0);
  
  glutInitWindowSize(1,1);
  window=glutCreateWindow(Prefix);
  glGetIntegerv(GL_MAX_VIEWPORT_DIMS, viewportLimit);
  glutDestroyWindow(window);

  Width=min(width,viewportLimit[0]);
  Height=min(height,viewportLimit[1]);
  
  glutInitWindowSize(Width,Height);
  window=glutCreateWindow(Prefix);
  
  if(settings::verbose > 1) 
    cout << "Rendering " << prefix << endl;
    
  glClearColor(1.0,1.0,1.0,0.0);
   
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_MAP1_VERTEX_3);
  glEnable(GL_MAP2_VERTEX_3);
  glEnable(GL_AUTO_NORMAL);
  
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glGetFloatv(GL_MODELVIEW_MATRIX,Rotate);
  glGetFloatv(GL_MODELVIEW_MATRIX,Modelview);
  
  initlights();
  
  glutReshapeFunc(reshape);
  glutDisplayFunc(display);
  glutKeyboardFunc(keyboard);
  glutMouseFunc(mouse);
  glutMouseWheelFunc(mousewheel);
   
  glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,GLUT_ACTION_CONTINUE_EXECUTION);

  glutCreateMenu(menu);
  glutAddMenuEntry("Home",1);
  glutAddMenuEntry("X spin",2);
  glutAddMenuEntry("Y spin",3);
  glutAddMenuEntry("Z spin",4);
  glutAddMenuEntry("Stop",5);
  glutAddMenuEntry("Export",6);
  glutAddMenuEntry("Quit",7);
  glutAttachMenu(GLUT_MIDDLE_BUTTON);

  glutMainLoop();
  
  width=Width;
  height=Height;
}
  
} // namespace gl

#endif
