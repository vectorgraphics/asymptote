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
  triple Min,Max;
public:
  drawPath3(path3 g, pen pentype) : g(g), pentype(pentype), controls(NULL) {}
    
  virtual ~drawPath3() {
    if(controls) delete controls;
  }

  bool is3D() {return true;}
  
  void bounds(bbox3& b) {
    Min=g.min();
    Max=g.max();
    b.add(Min);
    b.add(Max);
  }
  
  bool write(prcfile *out);
  
  bool render(int, double size2, const bbox3& b, bool transparent);

  drawElement *transformed(vm::array *t);
};

}

#endif
