/*****
 * drawpath3.h
 *
 * Stores a path3 that has been added to a picture.
 *****/

#ifndef DRAWPATH3_H
#define DRAWPATH3_H

#include "drawelement.h"
#include "path3.h"

namespace camp {

typedef double Triple[3];
  
class drawPath3 : public drawElement {
protected:
  const path3 g;
  bool straight;
  RGBAColour color;
  bool invisible;
  triple Min,Max;
  Triple *controls;
public:
  drawPath3(path3 g, const pen&p) :
    g(g), straight(g.piecewisestraight()), color(rgba(p)),
    invisible(p.invisible()), Min(g.min()), Max(g.max()), controls(NULL) {}
    
  drawPath3(const vm::array& t, const drawPath3 *s) :
    g(camp::transformed(t,s->g)), straight(s->straight), color(s->color),
    invisible(s->invisible), Min(g.min()), Max(g.max()), controls(NULL) {}
  
  virtual ~drawPath3() {
    if(controls) delete controls;
  }

  bool is3D() {return true;}
  
  void bounds(bbox3& B) {
    B.add(Min);
    B.add(Max);
  }
  
  void bounds(pair &b, double (*m)(double, double),
              double (*x)(const triple&, double*),
              double (*y)(const triple&, double*),
              double *t, bool &first) {
    pair z=g.bounds(m,x,y,t);
    if(first) {
      b=z;
      first=false;
    } else b=pair(m(b.getx(),z.getx()),m(b.gety(),z.gety()));
  }
  
  bool write(prcfile *out);
  
  void render(GLUnurbs*, double, const triple&, const triple&, double,
              bool transparent);

  drawElement *transformed(const vm::array& t);
};

}

#endif
