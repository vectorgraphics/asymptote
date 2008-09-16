/*****
 * drawsurface.cc
 *
 * Stores a surface that has been added to a picture.
 *****/

#include "drawsurface.h"
#include "path3.h"
#include <GL/glut.h>

namespace camp {

using vm::array;

void drawSurface::bounds(bbox3& b)
{
  double xmin,xmax;
  double ymin,ymax;
  double zmin,zmax;
  double c[16];
    
  for(int i=0; i < 16; ++i)
    c[i]=controls[i][0];
  bounds(xmin,xmax,c);
    
  for(int i=0; i < 16; ++i)
    c[i]=controls[i][1];
  bounds(ymin,ymax,c);
    
  for(int i=0; i < 16; ++i)
    c[i]=controls[i][2];
  bounds(zmin,zmax,c);
    
  Min=triple(xmin,ymin,zmin);
  Max=triple(xmax,ymax,zmax);
    
  b.add(Min);
  b.add(Max);
}
  
bool drawSurface::write(prcfile *out)
{
  if(invisible)
    return true;

  PRCMaterial m(ambient,diffuse,emissive,specular,opacity,shininess);
  out->add(new PRCBezierSurface(out,3,3,4,4,controls,m,granularity));
  
  return true;
}

// return a normal vector for the plane through u, v, and w.
inline triple normal(const Triple& u, const Triple& v, const Triple& w) 
{
  return cross(triple(v[0]-u[0],v[1]-u[1],v[2]-u[2]),
	       triple(w[0]-u[0],w[1]-u[1],w[2]-u[2]));
}

// return the perpendicular displacement of a point z from the plane
// through u with unit normal n.
inline triple displacement2(const Triple& z, const Triple& u, const triple& n)
{
  triple Z=triple(z[0]-u[0],z[1]-u[1],z[2]-u[2]);
  return n != triple(0,0,0) ? dot(Z,n)*n : Z;
}

inline double fraction(const Triple& z0, const Triple& c0,
		       const Triple& c1, const Triple& z1,
		       const triple& size3)
{
  triple Z0(z0[0],z0[1],z0[2]);
  triple Z1(z1[0],z1[1],z1[2]);
  return max(camp::fraction(displacement(triple(c0[0],c0[1],c0[2]),Z0,Z1),
			    size3),
	     camp::fraction(displacement(triple(c1[0],c1[1],c1[2]),Z0,Z1),
			    size3));
}

void drawSurface::fraction(double &F, const triple& size3)
{
  for(int i=0; i < 16; ++i) {
    Triple& C=controls[i];
    c[3*i]=C[0];
    c[3*i+1]=C[1];
    c[3*i+2]=C[2];
  }
  Triple& v0=controls[0];
  triple N=unit(normal(v0,controls[3],controls[15])+
		normal(v0,controls[15],controls[12]));
  f=0;
  for(int i=1; i < 16; ++i) 
    f=camp::max(f,camp::fraction(displacement2(controls[i],v0,N),size3));
  
  for(int i=0; i < 4; ++i)
    f=camp::max(f,camp::fraction(controls[4*i],controls[4*i+1],controls[4*i+2],
				 controls[4*i+3],size3));
  for(int i=0; i < 4; ++i)
    f=camp::max(f,camp::fraction(controls[i],controls[i+4],controls[i+8],
				 controls[i+12],size3));
  f=2*pixelfactor2*f;
  if(f > F) F=f;
}
  
bool drawSurface::render(int n, double size2, const bbox3& b, bool transparent)
{
  if(invisible || ((diffuse.A < 1.0) ^ transparent))
    return true;
  
  if(b.left > Max.getx() || b.right < Min.getx() || 
     b.bottom > Max.gety() || b.top < Min.gety() ||
     b.lower > Max.getz() || b.upper < Min.getz()) return true;
  
  GLfloat Diffuse[]={diffuse.R,diffuse.G,diffuse.B,diffuse.A};
  GLfloat Ambient[]={ambient.R,ambient.G,ambient.B,ambient.A};
  GLfloat Emissive[]={emissive.R,emissive.G,emissive.B,emissive.A};
  GLfloat Specular[]={specular.R,specular.G,specular.B,specular.A};
    
  glMaterialfv(GL_FRONT_AND_BACK,GL_DIFFUSE,Diffuse);
  glMaterialfv(GL_FRONT_AND_BACK,GL_AMBIENT,Ambient);
  glMaterialfv(GL_FRONT_AND_BACK,GL_EMISSION,Emissive);
  glMaterialfv(GL_FRONT_AND_BACK,GL_SPECULAR,Specular);
  glMaterialf(GL_FRONT_AND_BACK,GL_SHININESS,128.0*shininess);

  if(n >= settings::getSetting<Int>("threshold")) {
    static GLUnurbsObj *nurb=NULL;
    if(!nurb) {
      nurb=gluNewNurbsRenderer();
      gluNurbsProperty(nurb,GLU_SAMPLING_METHOD,GLU_PARAMETRIC_ERROR);
      gluNurbsProperty(nurb,GLU_PARAMETRIC_TOLERANCE,1.0);
      gluNurbsProperty(nurb,GLU_DISPLAY_MODE,GLU_FILL);
      gluNurbsProperty(nurb,GLU_CULLING,GLU_TRUE);
    }
  
    static GLfloat knots[8]={0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0};
    gluBeginSurface(nurb);
    gluNurbsSurface(nurb,8,knots,8,knots,3,12,(GLfloat*) &c,4,4,
		    GL_MAP2_VERTEX_3);
    gluEndSurface(nurb);
  } else {
    bool twosided=settings::getSetting<bool>("twosided");
    if(twosided) glFrontFace(GL_CW); // Work around GL_LIGHT_MODEL_TWO_SIDE bug.
    glMap2d(GL_MAP2_VERTEX_3,0,1,3,4,0,1,12,4,(GLdouble*) &controls);

    glMapGrid2d(n,0.0,1.0,n,0.0,1.0);
    glEvalMesh2(GL_FILL,0,n,0,n);
    if(twosided) glFrontFace(GL_CCW);
  }
  
  return true;
}

drawElement *drawSurface::transformed(array *t)
{
  return new drawSurface(t,this);
}
  
} //namespace camp
