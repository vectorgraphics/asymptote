/*****
 * drawpath3.cc
 *
 * Stores a path3 that has been added to a picture.
 *****/

#include "drawpath3.h"
#include "glrender.h"

namespace camp {

using vm::array;
  
inline void store(Triple& control, const triple& v)
{
  control[0]=v.getx();
  control[1]=v.gety();
  control[2]=v.getz();
}
  
#ifdef HAVE_LIBGLUT
inline void store(GLfloat *control, const triple& v)
{
  control[0]=v.getx();
  control[1]=v.gety();
  control[2]=v.getz();
}
#endif

bool drawPath3::write(prcfile *out)
{
  Int n=g.length();
  if(n == 0 || pentype.invisible())
    return true;

  RGBAColour color=rgba(pentype);
    
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
  double opacity=pentype.opacity();
  if(n == 0 || pentype.invisible() || ((opacity < 1.0) ^ transparent))
    return;

  pentype.torgb();
  glDisable(GL_LIGHTING);
  glColor4f(pentype.red(),pentype.green(),pentype.blue(),opacity);	

  if(straight) {
    controls=new Triple[n+1];
    glBegin(GL_LINE_STRIP);
    for(Int i=0; i <= n; ++i) {
      triple v=g.point(i);
      glVertex3f(v.getx(),v.gety(),v.getz());
    }
    glEnd();
  } else {
    for(Int i=0; i <= n; ++i) {
      static GLfloat knots[8]={0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0};
      static GLfloat controlpoints[12];
      store(controlpoints,g.point(i));
      store(controlpoints+3,g.postcontrol(i));
      store(controlpoints+6,g.precontrol(i+1));
      store(controlpoints+9,g.point(i+1));
    
      gluBeginCurve(nurb);
      gluNurbsCurve(nurb,8,knots,3,controlpoints,4,GL_MAP1_VERTEX_3);
      gluEndSurface(nurb);
    }
  }
  glEnable(GL_LIGHTING);

#endif
}

drawElement *drawPath3::transformed(array *t)
{
  return new drawPath3(camp::transformed(t,g),pentype);
}
  
} //namespace camp
