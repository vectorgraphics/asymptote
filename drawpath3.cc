/*****
 * drawpath3.cc
 *
 * Stores a path3 that has been added to a picture.
 *****/

#include <GL/glut.h>
#include "drawpath3.h"

namespace camp {

using vm::array;
  
inline void store(Triple& control, const triple& v)
{
  control[0]=v.getx();
  control[1]=v.gety();
  control[2]=v.getz();
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
    out->add(new PRCline(out,n+1,controls,color,scale3D));
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

bool drawPath3::render(GLUnurbsObj *, int, double size2, const bbox3& b,
		       bool transparent, bool)
{
  Int n=g.length();
  double opacity=pentype.opacity();
  if(n == 0 || pentype.invisible() || ((opacity < 1.0) ^ transparent) ||
     b.left > Max.getx() || b.right < Min.getx() || 
     b.bottom > Max.gety() || b.top < Min.gety() ||
     b.lower > Max.getz() || b.upper < Min.getz()) return true;
  
  triple size3=b.Max()-b.Min();
  
  pentype.torgb();
  glDisable(GL_LIGHTING);
  glColor4d(pentype.red(),pentype.green(),pentype.blue(),opacity);	

  if(g.piecewisestraight()) {
    controls=new Triple[n+1];
    glBegin(GL_LINE_STRIP);
    for(Int i=0; i <= n; ++i) {
      triple v=g.point(i);
      glVertex3d(v.getx(),v.gety(),v.getz());
    }
    glEnd();
  } else {
    for(Int i=0; i < n; ++i) {
      triple z0=g.point(i);
      triple c0=g.postcontrol(i);
      triple c1=g.precontrol(i+1);
      triple z1=g.point(i+1);
      double f=max(camp::fraction(displacement(c0,z0,z1),size3),
		     camp::fraction(displacement(c1,z0,z1),size3));
      int n=max(1,(int) (pixelfactor*f*size2+0.5));
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
  glEnable(GL_LIGHTING);

  return true;
}

drawElement *drawPath3::transformed(array *t)
{
  return new drawPath3(camp::transformed(t,g),pentype);
}
  
} //namespace camp
