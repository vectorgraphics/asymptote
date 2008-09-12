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

#ifdef HAVE_LIBGLUT
#include <GL/glut.h>
#include <GL/freeglut_ext.h>

namespace gl {
  
using camp::picture;
using camp::triple;
using vm::array;
using camp::scale3D;

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

bool Interactive;
bool Save;
const picture* Picture;
int Width=0;
int Height=0;
triple Max;
triple Min;
double miny;
double maxy;
double hy;
double cz;

double H;
unsigned char *Data;
bool first;  
GLint viewportLimit[2];
triple Light; 
double xmin,xmax;
double ymin,ymax;
  
double Zoom=1.0;
double X,Y;

const double moveFactor=1.0;  
const double zoomFactor=1.05;
const double zoomFactorStep=0.25;
const float rotateStep=0.25;
const float arcballRadius=500.0;

const double degrees=180.0/M_PI;
const double radians=1.0/degrees;

int window;
  
void initlights(void)
{
  GLfloat ambient[]={0.1, 0.1, 0.1, 1.0};
  GLfloat position[]={Light.getx(), Light.gety(), Light.getz(), 0.0};

  glLightModeli(GL_LIGHT_MODEL_TWO_SIDE,GL_TRUE);
  
  glEnable(GL_LIGHTING);
  glEnable(GL_LIGHT0);
  glLightfv(GL_LIGHT0,GL_AMBIENT,ambient);
  glLightfv(GL_LIGHT0,GL_POSITION,position);
}

void save()
{  
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  size_t ndata=3*Width*Height; // Use ColorComponents[colorspace]
  Data=new unsigned char[ndata];
  glReadPixels(0,0,Width,Height,GL_RGB,GL_UNSIGNED_BYTE,Data);
}
  
void quit() 
{
  if(Save) save();
  glutDestroyWindow(window);
  glutLeaveMainLoop();
}

void display(void)
{
  if(!Interactive && !first)
    return;
  
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // Render opaque objects
  Picture->render(Width,Height,Zoom,false);
  
  // Enable transparency
  glEnable(GL_BLEND);
  glDepthMask(GL_FALSE);
  glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
  
  // Render transparent objects
  Picture->render(Width,Height,Zoom,true);
  glDepthMask(GL_TRUE);
  glDisable(GL_BLEND);
  
  if(Interactive)
    glutSwapBuffers();
  else {
    glReadBuffer(GL_BACK_LEFT);
    quit();
  }
}

Arcball arcball;
  
void setProjection()
{
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  double Aspect=((double) Width)/Height;
  double X0=X*(xmax-xmin)/(Zoom*Width);
  double Y0=Y*(ymax-ymin)/(Zoom*Height);
  ymin=miny*Zoom-Y0;
  ymax=maxy*Zoom-Y0;
  double factor=Zoom*Aspect;
  xmin=-hy*factor-X0;
  xmax=hy*factor-X0;
  if(H == 0.0)
    glOrtho(xmin,xmax,ymin,ymax,-Max.getz(),-Min.getz());
  else {
    double r=H*Zoom;
    ymin=-r-Y0;
    ymax=r-Y0;
    xmin=-r*Aspect-X0;
    xmax=r*Aspect-X0;
    glFrustum(xmin,xmax,ymin,ymax,-Max.getz(),-Min.getz());
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
  case 17: // Ctrl-q
  case 'q':
    {
      glReadBuffer(GL_FRONT_LEFT);
      quit();
      break;
    }
  }
}
 
int x0,y0;
double lastzoom;
int mod;

double zangle;
double lastangle;

float Rotate[16];

void update() 
{
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glTranslatef(0,0,cz);
  glRotatef(zangle,0,0,1);
  glMultMatrixf(Rotate);
  glTranslatef(0,0,-cz);
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
  
void zoom(int x, int y)
{
  static const double limit=log(0.1*DBL_MAX)/log(zoomFactor);
  if(x > 0 && y > 0) {
    double s=zoomFactorStep*(y0-y);
    if(fabs(s) < limit) {
      Zoom=lastzoom*pow(zoomFactor,s);
      setProjection();
      glutPostRedisplay();
    }
  }
}
  
void mousewheel(int wheel, int direction, int x, int y) 
{
  if(direction > 0)
    Zoom /= zoomFactor;
  else 
    Zoom *= zoomFactor;
  setProjection();
  glutPostRedisplay();
}

void rotate(int x, int y)
{
  arcball.mouse_motion(x,Height-y,0,
		       mod == GLUT_ACTIVE_SHIFT, // X rotation only
		       mod == GLUT_ACTIVE_ALT);  // Y rotation only

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

void rotateZ(int x, int y)
{
  double angle=Degrees(x,y);
  zangle += angle-lastangle;
  lastangle=angle;
  update();
}

void mouse(int button, int state, int x, int y)
{
  mod=glutGetModifiers();
  if(button == GLUT_LEFT_BUTTON) {
    if(mod == GLUT_ACTIVE_CTRL) {
      if(state == GLUT_DOWN) {
	x0=x; y0=y;
	glutMotionFunc(move);
      }
    } else {
      if(state == GLUT_DOWN) {
	arcball.mouse_down(x,Height-y);
	glutMotionFunc(rotate);
      } else
	arcball.mouse_up();
    }
  } else if(button == GLUT_RIGHT_BUTTON) {
    if(mod == GLUT_ACTIVE_SHIFT) {
      lastangle=Degrees(x,y);
      glutMotionFunc(rotateZ);
    } else {
      if(state == GLUT_DOWN) {
	x0=x; y0=y;
	lastzoom=Zoom;
	glutMotionFunc(zoom);
      }
    }	
  }
}

// angle=0 means orthographic.
void glrender(const char *prefix, unsigned char* &data,  const picture *pic,
	      int& width, int& height, const triple& light,
	      double angle, const triple& m, const triple& M,
	      bool interactive, bool save)
{
  Interactive=interactive;
  Save=save;
  Picture=pic;
  Light=light;
  Min=m;
  Max=M;
  miny=Min.gety();
  maxy=Max.gety();
  hy=0.5*(maxy-miny);
  cz=0.5*(Min.getz()+Max.getz());
  H=angle != 0.0 ? -tan(0.5*angle*radians)*Max.getz() : 0.0;
  first=true;
   
  X=Y=0.0;
  
  string options=string(settings::argv0)+" ";
  if(!Interactive) options += "-iconic ";
  options += settings::getSetting<string>("glOptions");
  char **argv=args(options.c_str(),true);
  int argc=0;
  while(argv[argc] != NULL)
    ++argc;
  glutInit(&argc,argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
  glutInitWindowPosition(0,0);
  
  glutInitWindowSize(1,1);
  window=glutCreateWindow(prefix);
  glGetIntegerv(GL_MAX_VIEWPORT_DIMS, viewportLimit);
  glutDestroyWindow(window);

  Width=min(width,viewportLimit[0]);
  Height=min(height,viewportLimit[1]);
  
  glutInitWindowSize(Width,Height);
  window=glutCreateWindow(prefix);
  
  if(settings::verbose > 1) 
    cout << "Rendering " << prefix << endl;
    
  glClearColor(1.0,1.0,1.0,0.0);
   
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_MAP1_VERTEX_3);
  glEnable(GL_MAP2_VERTEX_3);
  glEnable(GL_AUTO_NORMAL);
  
  // GL_LIGHT_MODEL_TWO_SIDE seems to require CW orientation.  
  glFrontFace(GL_CW);
  
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glGetFloatv(GL_MODELVIEW_MATRIX,Rotate);
  
  initlights();
  glutReshapeFunc(reshape);
  glutDisplayFunc(display);
  glutKeyboardFunc(keyboard);
  glutMouseFunc(mouse);
  glutMouseWheelFunc(mousewheel);
   
  glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,GLUT_ACTION_CONTINUE_EXECUTION);
  Data=NULL;
  glutMainLoop();
  width=Width;
  height=Height;
  data=Data;
}
  
} // namespace gl

#endif
