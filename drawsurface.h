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
  triple normal;
  
  bool invisible;
  triple Min,Max;
  static triple c3[16];
  GLfloat *colors;
  bool havecolors;
  
#ifdef HAVE_LIBGLUT
  GLfloat c[48];
  triple d; // Maximum deviation of surface from a quadrilateral.
  triple dperp;
  GLfloat v1[16];
  GLfloat v2[16];
  GLfloat Normal[3];
  bool havenormal;
  bool havetransparency;
#endif  
  
  void storecolor(int i, const vm::array &pens, int j) {
    pen p=vm::read<camp::pen>(pens,j);
    p.torgb();
    colors[i]=p.red();
    colors[i+1]=p.green();
    colors[i+2]=p.blue();
    colors[i+3]=p.opacity();
  }
  
public:
  drawSurface(const vm::array& g, bool straight, const vm::array&p,
	      double opacity, double shininess, double granularity,
	      triple normal, const vm::array &pens) :
    straight(straight), opacity(opacity), shininess(shininess),
    granularity(granularity), normal(unit(normal)) {
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
    
#ifdef HAVE_LIBGLUT
    int size=checkArray(&pens);
    havecolors=size > 0;
    if(havecolors) {
      colors=new GLfloat[16];
      if(size != 4) reportError(wrongsize);
      storecolor(0,pens,0);
      storecolor(4,pens,1);
      storecolor(12,pens,2);
      storecolor(8,pens,3);
    }
#endif    
    
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
    shininess(s->shininess), granularity(s->granularity),
    invisible(s->invisible), colors(s->colors), havecolors(s->havecolors) {
    for(size_t i=0; i < 16; ++i) {
      const double *c=s->controls[i];
      triple v=run::operator *(t,triple(c[0],c[1],c[2]));
      controls[i][0]=v.getx();
      controls[i][1]=v.gety();
      controls[i][2]=v.getz();
    }
    normal=run::multshiftless(t,s->normal);
  }
  
  bool is3D() {return true;}
  
  void bounds(bbox3& b);
  
  void bounds(pair &b, double (*m)(double, double),
	      double (*x)(const triple&, double*),
	      double (*y)(const triple&, double*),
	      double *t, bool &first);
  
  virtual ~drawSurface() {
    if(havecolors)
      delete[] colors;
  }

  bool write(prcfile *out);
  
  void displacement();
  void render(GLUnurbs *nurb, double, const triple& Min, const triple& Max,
	      double perspective, bool transparent, bool twosided);
  
  drawElement *transformed(const vm::array& t);
};
  
}

#endif
