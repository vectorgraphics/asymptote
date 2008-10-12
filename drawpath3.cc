/*****
 * drawpath3.cc
 *
 * Stores a path3 that has been added to a picture.
 *****/

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
  if(n == 0 || invisible)
    return true;

  if(straight) {
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

void drawPath3::render(GLUnurbs *nurb, double, const triple&, const triple&,
		       double, bool transparent, bool)
{
#ifdef HAVE_LIBGLUT
  Int n=g.length();
  if(n == 0 || invisible || ((color.A < 1.0) ^ transparent))
    return;

  bool lightEnabled=glIsEnabled(GL_LIGHTING);
  if(lightEnabled)
    glDisable(GL_LIGHTING);
  
  glColor4f(color.R,color.G,color.B,color.A);
      
  if(straight) {
    glBegin(GL_LINE_STRIP);
    for(Int i=0; i <= n; ++i) {
      triple v=g.point(i);
      glVertex3f(v.getx(),v.gety(),v.getz());
    }
    glEnd();
  } else {
    for(Int i=0; i < n; ++i) {
      static GLfloat knots[8]={0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0};
      static GLfloat controlpoints[12];
      store(controlpoints,g.point(i));
      store(controlpoints+3,g.postcontrol(i));
      store(controlpoints+6,g.precontrol(i+1));
      store(controlpoints+9,g.point(i+1));
    
      gluBeginCurve(nurb);
      gluNurbsCurve(nurb,8,knots,3,controlpoints,4,GL_MAP1_VERTEX_3);
      gluEndCurve(nurb);
    }
  }
  if(lightEnabled) 
    glEnable(GL_LIGHTING);
  else 
    glColor4f(0.0,0.0,0.0,1.0);
#endif
}

drawElement *drawPath3::transformed(const array& t)
{
  return new drawPath3(t,this);
}
  
} //namespace camp
