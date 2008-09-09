/*****
 * drawsurface.h
 *
 * Stores a surface that has been added to a picture.
 *****/

#ifndef DRAWSURFACE_H
#define DRAWSURFACE_H

#include "drawelement.h"
#include "triple.h"
#include "arrayop.h"

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
  bool localsub; // Use minimum of local and global number of subdivisions
  triple min,max;
  Triple controls[16];
  bool invisible;
  double f; // Fraction of 3D bounding box occupied by surface.

public:
  drawSurface(const vm::array& g, const vm::array&p, double opacity,
	      double shininess, double granularity, bool localsub, 
	      triple min, triple max) :
    opacity(opacity), shininess(shininess), granularity(granularity),
    localsub(localsub), min(min), max(max) {
    
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
    for(size_t i=0; i < 4; ++i) {
      vm::array *gi=vm::read<vm::array*>(g,i);
      size_t gisize=checkArray(gi);
      if(gisize != 4) 
	reportError(wrongsize);
      for(size_t j=0; j < 4; ++j) {
	triple v=vm::read<triple>(gi,j);
	controls[k][0]=v.getx()*scale3D;
	controls[k][1]=v.gety()*scale3D;
	controls[k][2]=v.getz()*scale3D;
	++k;
      }
    }
  }
  
  drawSurface(vm::array *t, const drawSurface *s) :
    diffuse(s->diffuse), ambient(s->ambient), emissive(s->emissive),
    specular(s->specular), opacity(s->opacity), shininess(s->shininess),
    granularity(s->granularity), localsub(s->localsub), 
    invisible(s->invisible) {
    min=run::operator *(t,s->min); // TODO: Bounds needs to be recalculated.
    max=run::operator *(t,s->max);
    for(size_t i=0; i < 16; ++i) {
      const double *c=s->controls[i];
      triple v=run::operator *(t,triple(c[0]/scale3D,c[1]/scale3D,c[2]/scale3D));
      controls[i][0]=v.getx()*scale3D;
      controls[i][1]=v.gety()*scale3D;
      controls[i][2]=v.getz()*scale3D;
    }
  }
  
  bool is3D() {return true;}
  
  void bounds(bbox3& b) {
    b.add(min);
    b.add(max);
  }
  
  virtual ~drawSurface() {}

  bool write(prcfile *out);
  
  void fraction(double &f, const triple& size3);
  bool render(int n, double, const triple&, bool transparent);
  
  drawElement *transformed(vm::array *t);
};
  
}

#endif
