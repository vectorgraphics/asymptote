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
#include "path3.h"

namespace camp {

typedef double Triple[3];
  
class drawSurface : public drawElement {
protected:
  Triple controls[16];
  bool straight;
  RGBAColour diffuse;
  RGBAColour ambient;
  RGBAColour emissive;
  RGBAColour specular;
  double opacity;
  double shininess;
  double granularity;
  bool lighton;
  
  bool invisible;
  bool degenerate;
  triple Min,Max;
#ifdef HAVE_LIBGLUT
  GLfloat c[48];
  GLfloat d[12];
  double f; // Fraction of 3D bounding box occupied by surface.
  GLfloat v1[16];
  GLfloat v2[16];
#endif  
  
public:
  drawSurface(const vm::array& g, bool straight, const vm::array&p,
	      double opacity, double shininess, double granularity,
	      bool lighton) : 
    straight(straight), opacity(opacity), shininess(shininess),
    granularity(granularity), lighton(lighton) {
    
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
	controls[k][0]=v.getx();
	controls[k][1]=v.gety();
	controls[k][2]=v.getz();
	++k;
      }
    }
  }
  
  drawSurface(const vm::array& t, const drawSurface *s) :
    straight(s->straight), diffuse(s->diffuse), ambient(s->ambient),
    emissive(s->emissive), specular(s->specular), opacity(s->opacity),
    shininess(s->shininess), granularity(s->granularity), lighton(s->lighton),
    invisible(s->invisible) {
    for(size_t i=0; i < 16; ++i) {
      const double *c=s->controls[i];
      triple v=run::operator *(t,triple(c[0],c[1],c[2]));
      controls[i][0]=v.getx();
      controls[i][1]=v.gety();
      controls[i][2]=v.getz();
    }
  }
  
  bool is3D() {return true;}
  
  void bounds(double &Min, double &Max, double *c) {
    Min=bound(c,min,c[0]);
    Max=bound(c,max,c[0]);
  }
  
  void bounds(bbox3& b);
  
  virtual ~drawSurface() {}

  bool write(prcfile *out);
  
  triple normal(const Triple& u, const Triple& v, const Triple& w);
  
  void fraction(const triple& size3);
  void render(GLUnurbs *nurb, double, const triple& Min, const triple& Max,
	      double perspective, bool transparent, bool twosided);
  
  drawElement *transformed(const vm::array& t);
};
  
}

#endif
