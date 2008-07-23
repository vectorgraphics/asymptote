/*****
 * drawpath3.h
 *
 * Stores a path3 that has been added to a picture.
 *****/

#ifndef DRAWPATH3_H
#define DRAWPATH3_H

#include "drawelement.h"
#include "triple.h"

namespace camp {

typedef double Triple[3];
  
class drawPath3 : public drawElement {
protected:
  pen pentype;
  bool straight;
  triple min,max;
  size_t n;
  Triple *controls;
public:
  drawPath3(const vm::array& g, pen pentype, bool straight, triple min,
	    triple max) : 
    pentype(pentype), straight(straight), min(min), max(max) {
    n=checkArray(&g);
    controls=new Triple[n];
  
    const double factor=1.0/settings::cm;
    for(size_t i=0; i < n; ++i) {
      triple v=vm::read<triple>(g,i);
      controls[i][0]=v.getx()*factor;
      controls[i][1]=v.gety()*factor;
      controls[i][2]=v.getz()*factor;
    }
  }
  
  virtual ~drawPath3() {
    delete *controls;
  }

  bool is3D() {return true;}
  
  void bounds(bbox3& b) {
    b.add(min);
    b.add(max);
  }
  
  bool write(prcfile *out);
};

}

#endif
