/*****
 * drawpath3.cc
 *
 * Stores a path3 that has been added to a picture.
 *****/

#include "drawpath3.h"

namespace camp {

inline void store(Triple& control, const triple& v)
{
  static const double factor=1.0/settings::cm;
  control[0]=v.getx()*factor;
  control[1]=v.gety()*factor;
  control[2]=v.getz()*factor;
}
  
bool drawPath3::write(prcfile *out)
{
  if(g.length() == 0 || pentype.invisible())
    return true;

  RGBAColour color=rgba(pentype);
    
  if(g.piecewisestraight()) {
    Int n=g.size();
    controls=new Triple[n];
    for(Int i=0; i < n; ++i)
      store(controls[i],g.point(i));
    out->add(new PRCline(out,n,controls,color));
  } else {
    Int n=g.length();
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

} //namespace camp
