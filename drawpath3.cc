/*****
 * drawpath3.cc
 *
 * Stores a path3 that has been added to a picture.
 *****/

#include <GL/glut.h>
#include "drawpath3.h"

namespace camp {

using vm::array;
  
static const double factor=1.0/settings::cm;
  
inline void store(Triple& control, const triple& v)
{
  control[0]=v.getx()*factor;
  control[1]=v.gety()*factor;
  control[2]=v.getz()*factor;
}
  
bool drawPath3::write(prcfile *out)
{
  Int n=g.length();
  if(n == 0 || pentype.invisible())
    return true;

  RGBAColour color=rgba(pentype);
    
  if(g.piecewisestraight()) {
    controls=new Triple[n+1];
    for(Int i=0; i <= n; ++i)
      store(controls[i],g.point(i));
    out->add(new PRCline(out,n+1,controls,color));
  } else {
    int m=3*n+1;
    controls=new Triple[m];
    store(controls[0],g.point((Int) 0));
    store(controls[1],g.postcontrol((Int) 0));
    size_t k=1;
    for(Int i=1; i < n; ++i) {
      store(controls[++k],g.precontrol(i));
      store(controls[++k],g.point(i));
      store(controls[++k],g.postcontrol(i));
    }
    store(controls[++k],g.precontrol((Int) n));
    store(controls[++k],g.point((Int) n));
    out->add(new PRCBezierCurve(out,3,m,controls,color));
  }

  return true;
}

bool drawPath3::render(int, double size2, const triple& size3)
{
  Int n=g.length();
  if(n == 0 || pentype.invisible())
    return true;

  GLfloat p[]={pentype.red(),pentype.green(),pentype.blue(),pentype.opacity()};
  static GLfloat black[] = {0.0,0.0,0.0,1.0};
  glMaterialfv(GL_FRONT,GL_DIFFUSE,p);
  glMaterialfv(GL_FRONT,GL_AMBIENT,black);
  glMaterialfv(GL_FRONT,GL_EMISSION,p);
  glMaterialfv(GL_FRONT,GL_SPECULAR,black);
  glMaterialf(GL_FRONT,GL_SHININESS,100.0);
    
  if(g.piecewisestraight()) {
    controls=new Triple[n+1];
    glBegin(GL_LINE_STRIP);
    for(Int i=0; i <= n; ++i) {
      triple v=g.point(i);
      glVertex3d(v.getx()*factor,v.gety()*factor,v.getz()*factor);
    }
    glEnd();
  } else {
    for(Int i=0; i < n; ++i) {
      triple z0=g.point(i)*scale3D;
      triple c0=g.postcontrol(i)*scale3D;
      triple c1=g.precontrol(i+1)*scale3D;
      triple z1=g.point(i+1)*scale3D;
      double f=max(camp::fraction(displacement(c0,z0,z1),size3),
		   camp::fraction(displacement(c1,z0,z1),size3));
      int n=max(1,(int) ceil(pixelfactor*f*size2));
      triple controls[]={z0,c0,c1,z1};
      GLdouble controlpoints[12];
      for(size_t i=0; i < 4; ++i) {
	triple& c=controls[i];
	size_t i3=3*i;
	controlpoints[i3]=c.getx();
	controlpoints[i3+1]=c.gety();
	controlpoints[i3+2]=c.getz();
      }
      glMap1d(GL_MAP1_VERTEX_3,0.0,1.0,3,4,controlpoints);
      glMapGrid1d(n,0.0,1.0);
      glEvalMesh1(GL_LINE,0,n);
    }
  }
  return true;
}

drawElement *drawPath3::transformed(array *t)
{
  return new drawPath3(camp::transformed(t,g),pentype);
}
  
} //namespace camp
