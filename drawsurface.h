/*****
 * drawsurface.h
 *
 * Stores a surface that has been added to a picture.
 *****/

#ifndef DRAWSURFACE_H
#define DRAWSURFACE_H

#include "drawelement.h"
#include "triple.h"

namespace camp {

typedef double Triple[3];
  
class drawSurface : public drawElement {
protected:
  pen pentype;
  Triple controls[16];
public:
  drawSurface(const vm::array& g, pen pentype) : pentype(pentype) {
  
    int k=0;
    size_t gsize=checkArray(&g);
    string wrongsize="Bezier surface patch requires 4x4 array of triples";
    if(gsize != 4) 
      reportError(wrongsize);
    const double factor=1.0/settings::cm;
    for(size_t i=0; i < 4; ++i) {
      vm::array *gi=vm::read<vm::array*>(g,i);
      size_t gisize=checkArray(gi);
      if(gisize != 4) 
	reportError(wrongsize);
      for(size_t j=0; j < 4; ++j) {
	triple v=vm::read<triple>(gi,j);
	controls[k][0]=v.getx()*factor;
	controls[k][1]=v.gety()*factor;
	controls[k][2]=v.getz()*factor;
	++k;
      }
    }
  }
  
  virtual ~drawSurface() {}

  bool write(prcfile *out);
};

}

#endif
