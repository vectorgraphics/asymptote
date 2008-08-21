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
  RGBAColour diffuse;
  RGBAColour ambient;
  RGBAColour emissive;
  RGBAColour specular;
  double opacity;
  double shininess;
  double granularity;
  triple min,max;
  Triple controls[16];
  bool invisible;
public:
  drawSurface(const vm::array& g, const vm::array&p, double opacity,
	      double shininess, double granularity, triple min, triple max) :
    opacity(opacity), shininess(shininess), granularity(granularity),
    min(min), max(max) {
    
    string wrongsize=
      "Bezier surface patch requires 4x4 array of triples and array of 4 pens";
    if(checkArray(&g) != 4 || checkArray(&p) != 4)
      reportError(wrongsize);
    
    pen surfacepen=vm::read<camp::pen>(p,0);
    invisible=surfacepen.invisible();
    
    diffuse=rgba(surfacepen);
    ambient=rgba(vm::read<camp::pen>(p,1));
    emissive=rgba(vm::read<camp::pen>(p,2));
    specular=rgba(vm::read<camp::pen>(p,3));
    
    size_t k=0;
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
  
  bool is3D() {return true;}
  
  void bounds(bbox3& b) {
    b.add(min);
    b.add(max);
  }
  
  virtual ~drawSurface() {}

  bool write(prcfile *out);
};
  
}

#endif
