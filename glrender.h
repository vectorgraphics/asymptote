/*****
 * glrender.h
 * Render 3D Bezier paths and surfaces.
 *****/

#ifndef GLRENDER_H
#define GLRENDER_H

#include "common.h"
#include "triple.h"

#ifdef HAVE_GL

#include <csignal>

#define GLEW_NO_GLU
//#define GLEW_OSMESA

#ifdef __MSDOS__
#define GLEW_STATIC
#define _WIN32
#endif

#include "GL/glew.h"

#ifdef __APPLE__
#define GL_SILENCE_DEPRECATION
#include <OpenGL/gl.h>
#ifdef HAVE_LIBGLUT
#include <GLUT/glut.h>
#endif
#ifdef HAVE_LIBOSMESA
#include <GL/osmesa.h>
#endif
#else
#ifdef __MSDOS__
#undef _WIN32
#include <GL/gl.h>
#include <GL/wglew.h>
#include <GL/wglext.h>
#endif
#ifdef HAVE_LIBGLUT
#include <GL/glut.h>
#endif
#ifdef HAVE_LIBOSMESA
#include <GL/osmesa.h>
#endif
#endif

#else
typedef float GLfloat;
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

inline void store(GLfloat *control, const triple& v, double weight)
{
  control[0]=v.getx()*weight;
  control[1]=v.gety()*weight;
  control[2]=v.getz()*weight;
  control[3]=weight;
}

}

namespace gl {

extern bool outlinemode;
extern bool wireframeMode;
extern size_t maxvertices;
extern bool forceRemesh;

extern bool orthographic;
extern double xmin,xmax;
extern double ymin,ymax;
extern double zmin,zmax;
extern int fullWidth,fullHeight;
extern double Zoom0;
extern double Angle;

extern GLuint ubo;

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

#ifdef HAVE_GL
GLuint initHDR();
#endif

projection camera(bool user=true);

void glrender(const string& prefix, const camp::picture* pic,
              const string& format, double width, double height, double angle,
              double zoom, const camp::triple& m, const camp::triple& M,
              const camp::pair& shift, double *t, double *background,
              size_t nlights, camp::triple *lights, double *diffuse,
              double *specular, bool view, int oldpid=0);

struct ModelView {
  double T[16];
  double Tinv[16];
};

extern ModelView modelView;

void initshader();
void deleteshader();

extern size_t Ncenter;
}

namespace camp {

extern GLint materialShader;
extern GLint colorShader;
extern GLint noNormalShader;
extern GLint pixelShader;

void setUniforms(GLint shader);
void deleteUniforms();

}

#endif
