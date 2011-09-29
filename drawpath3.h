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

class drawPath3 : public drawElement {
protected:
  const path3 g;
  triple center;
  bool straight;
  RGBAColour color;
  bool invisible;
  Interaction interaction;
  triple Min,Max;
public:
  drawPath3(path3 g, triple center, const pen& p, Interaction interaction) :
    g(g), center(center), straight(g.piecewisestraight()), color(rgba(p)),
    invisible(p.invisible()), interaction(interaction),
    Min(g.min()), Max(g.max()) {}
    
  drawPath3(const vm::array& t, const drawPath3 *s) :
    g(camp::transformed(t,s->g)), straight(s->straight),
    color(s->color), invisible(s->invisible), interaction(s->interaction),
    Min(g.min()), Max(g.max()) {
    center=run::operator *(t,s->center);
  }
  
  virtual ~drawPath3() {}

  bool is3D() {return true;}
  
  void bounds(bbox3& B) {
    B.add(Min);
    B.add(Max);
  }
  
  void ratio(pair &b, double (*m)(double, double), double, bool &first) {
    pair z=g.ratio(m);
    if(first) {
      b=z;
      first=false;
    } else b=pair(m(b.getx(),z.getx()),m(b.gety(),z.gety()));
  }
  
  bool write(prcfile *out, unsigned int *, vm::array *, vm::array *, double,
             groupsmap&);
  
  void render(GLUnurbs*, double, const triple&, const triple&, double,
              bool transparent);

  drawElement *transformed(const vm::array& t);
};

class drawNurbsPath3 : public drawElement {
protected:
  size_t degree;
  size_t n;
  Triple *controls;
  double *weights;
  double *knots;
  RGBAColour color;
  bool invisible;
  triple Min,Max;
  
#ifdef HAVE_GL
  GLfloat *Controls;
  GLfloat *Knots;
#endif  
  
public:
  drawNurbsPath3(const vm::array& g, const vm::array* knot,
                 const vm::array* weight, const pen& p) :
    color(rgba(p)), invisible(p.invisible()) {
    size_t weightsize=checkArray(weight);
    
    string wrongsize="Inconsistent NURBS data";
    n=checkArray(&g);
    
    if(n == 0 || (weightsize != 0 && weightsize != n))
      reportError(wrongsize);
    
    controls=new(UseGC) Triple[n];
    
    size_t k=0;
    for(size_t i=0; i < n; ++i)
      store(controls[k++],vm::read<triple>(g,i));
      
    if(weightsize > 0) {
      size_t k=0;
      weights=new(UseGC) double[n];
      for(size_t i=0; i < n; ++i)
        weights[k++]=vm::read<double>(weight,i);
    } else weights=NULL;
      
    size_t nknots=checkArray(knot);
    
    if(nknots <= n+1 || nknots > 2*n)
      reportError(wrongsize);

    degree=nknots-n-1;
    
    run::copyArrayC(knots,knot,0,NoGC);
    
#ifdef HAVE_GL
    Controls=NULL;
#endif  
  }
  
  drawNurbsPath3(const vm::array& t, const drawNurbsPath3 *s) :
    degree(s->degree), n(s->n), color(s->color), invisible(s->invisible) {
    controls=new(UseGC) Triple[n];
      
    for(size_t i=0; i < n; ++i) {
      const double *c=s->controls[i];
      triple v=run::operator *(t,triple(c[0],c[1],c[2]));
      controls[i][0]=v.getx();
      controls[i][1]=v.gety();
      controls[i][2]=v.getz();
    }
    
    if(s->weights) {
      weights=new(UseGC) double[n];
      for(size_t i=0; i < n; ++i)
        weights[i]=s->weights[i];
    } else weights=NULL;
    
    size_t nknots=degree+n+1;
    knots=new(UseGC) double[nknots];
    
    for(size_t i=0; i < nknots; ++i)
      knots[i]=s->knots[i];
    
#ifdef HAVE_GL
    Controls=NULL;
#endif    
  }
  
  bool is3D() {return true;}
  
  void bounds(bbox3& b);
  
  virtual ~drawNurbsPath3() {}

  bool write(prcfile *out, unsigned int *, vm::array *, vm::array *, double,
             groupsmap&);
  
  void displacement();
  void ratio(pair &b, double (*m)(double, double), double fuzz, bool &first);
    
  void render(GLUnurbs *nurb, double size2,
              const triple& Min, const triple& Max,
              double perspective, bool transparent);
    
  drawElement *transformed(const vm::array& t);
};

}

#endif
