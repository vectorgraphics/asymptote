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
public:
  drawPath3(path3 g, pen pentype) : g(g), pentype(pentype), controls(NULL) {}
    
  virtual ~drawPath3() {
    if(controls) delete controls;
  }

  bool is3D() {return true;}
  
  void bounds(bbox3& b) {
    b.add(g.min());
    b.add(g.max());
  }
  
  bool write(prcfile *out);
  
  bool render(int, double size2, const triple& size3, bool transparent);

  drawElement *transformed(vm::array *t);
};

}

#endif
