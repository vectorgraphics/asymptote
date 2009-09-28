/*****
 * glrender.h
 * Render 3D Bezier paths and surfaces.
 *****/

#ifndef GLRENDER_H
#define GLRENDER_H

#include "common.h"
#include "triple.h"

#ifdef HAVE_LIBGL

#include <csignal>

#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <OpenGL/glext.h>
#include <OpenGL/glu.h>
#include <GLUT/glut.h>
#ifdef GLU_TESS_CALLBACK_TRIPLEDOT
typedef GLvoid (* _GLUfuncptr)(...);
#else
typedef GLvoid (* _GLUfuncptr)();
#endif
#else
#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glu.h>
#include <GL/glut.h>
#endif

namespace camp {
class picture;

inline void store(GLfloat *f, double *C)
{
  f[0]=C[0];
  f[1]=C[1];
  f[2]=C[2];
}

inline void store(GLfloat *control, const camp::triple& v)
{
  control[0]=v.getx();
  control[1]=v.gety();
  control[2]=v.getz();
}
}

namespace gl {

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

projection camera(bool user=true);

void glrender(const string& prefix, const camp::picture* pic,
              const string& format, double width, double height, double angle,
              double zoom, const camp::triple& m, const camp::triple& M,
              const camp::pair& shift, double *t, double *background,
              size_t nlights, camp::triple *lights, double *diffuse,
              double *ambient, double *specular, bool viewportlighting,
              bool view, int oldpid=0);
}

#else
typedef void GLUnurbs;
typedef float GLfloat;
#endif

#endif


