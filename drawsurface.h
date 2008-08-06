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
  pen diffusepen;
  pen ambientpen;
  pen emissivepen;
  pen specularpen;
  double alpha;
  double shininess;
  triple min,max;
  Triple controls[16];
public:
  drawSurface(const vm::array& g, pen diffusepen_, pen ambientpen_,
	      pen emissivepen_, pen specularpen_, double alpha,
	      double shininess, triple min, triple max) :
    diffusepen(diffusepen_), ambientpen(ambientpen_),
    emissivepen(emissivepen_), specularpen(specularpen_), alpha(alpha),
    shininess(shininess), min(min), max(max) {
    
    diffusepen.torgb();
    ambientpen.torgb();
    emissivepen.torgb();
    specularpen.torgb();
    
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
