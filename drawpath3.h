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
  path3 g;
  pen pentype;
  Triple *controls;
  bool straight;
public:
  drawPath3(path3 g, pen pentype) : g(g), pentype(pentype), controls(NULL),
				    straight(g.piecewisestraight()) {}
    
  virtual ~drawPath3() {
    if(controls) delete controls;
  }

  bool is3D() {return true;}
  
  void bounds(bbox3& B) {
    B.add(g.min());
    B.add(g.max());
  }
  
  bool write(prcfile *out);
  
  void render(GLUnurbs*, double, const triple&, const triple&, double,
	      bool transparent, bool);

  drawElement *transformed(vm::array *t);
};

}

#endif
