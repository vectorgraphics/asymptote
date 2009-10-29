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
  bool straight;
  RGBAColour color;
  bool invisible;
  triple Min,Max;
  Triple *controls;
  string name;
public:
  drawPath3(path3 g, const pen& p, const string& name) :
    g(g), straight(g.piecewisestraight()), color(rgba(p)),
    invisible(p.invisible()), Min(g.min()), Max(g.max()), controls(NULL),
    name(name) {}
    
  drawPath3(const vm::array& t, const drawPath3 *s) :
    g(camp::transformed(t,s->g)), straight(s->straight), color(s->color),
    invisible(s->invisible), Min(g.min()), Max(g.max()), controls(NULL),
    name(s->name) {}
  
  virtual ~drawPath3() {}

  bool is3D() {return true;}
  
  void bounds(bbox3& B) {
    B.add(Min);
    B.add(Max);
  }
  
  void ratio(pair &b, double (*m)(double, double), bool &first) {
    pair z=g.ratio(m);
    if(first) {
      b=z;
      first=false;
    } else b=pair(m(b.getx(),z.getx()),m(b.gety(),z.gety()));
  }
  
  bool write(prcfile *out, unsigned int *count, vm::array *index,
             vm::array *origin);
  
  void render(GLUnurbs*, double, const triple&, const triple&, double,
              bool transparent);

  drawElement *transformed(const vm::array& t);
};

class drawNurbsPath3 : public drawElement {
protected:
  size_t udegree;
  size_t nu;
  Triple *controls;
  double *weights;
  double *uknots;
  RGBAColour color;
  bool invisible;
  string name;
  triple Min,Max;
  
#ifdef HAVE_LIBGL
  GLfloat *Controls;
  GLfloat *uKnots;
#endif  
  
public:
  drawNurbsPath3(const vm::array& g, const vm::array* uknot,
                 const vm::array* weight, const pen& p, const string& name) :
    color(rgba(p)), invisible(p.invisible()), name(name) {
    size_t weightsize=checkArray(weight);
    
    string wrongsize="Inconsistent NURBS data";
    nu=checkArray(&g);
    
    if(nu == 0 || (weightsize != 0 && weightsize != nu))
      reportError(wrongsize);
    
    controls=new(UseGC) Triple[nu];
    
    size_t k=0;
    for(size_t i=0; i < nu; ++i)
      store(controls[k++],vm::read<triple>(g,i));
      
    if(weightsize > 0) {
      size_t k=0;
      weights=new(UseGC) double[nu];
      for(size_t i=0; i < nu; ++i)
        weights[k++]=vm::read<double>(weight,i);
    } else weights=NULL;
      
    size_t nuknots=checkArray(uknot);
    
    if(nuknots <= nu+1 || nuknots > 2*nu)
      reportError(wrongsize);

    udegree=nuknots-nu-1;
    
    uknots=run::copyArrayC(uknot,0,NoGC);
    
#ifdef HAVE_LIBGL
    uKnots=new(UseGC) GLfloat[nuknots];
    Controls=new(UseGC) GLfloat[(weights ? 4 : 3)*nu];
#endif  
  }
  
  drawNurbsPath3(const vm::array& t, const drawNurbsPath3 *s) :
    udegree(s->udegree), nu(s->nu), color(s->color), invisible(s->invisible),
    name(s->name) {
    
    controls=new(UseGC) Triple[nu];
      
    for(size_t i=0; i < nu; ++i) {
      const double *c=s->controls[i];
      triple v=run::operator *(t,triple(c[0],c[1],c[2]));
      controls[i][0]=v.getx();
      controls[i][1]=v.gety();
      controls[i][2]=v.getz();
    }
    
    if(s->weights) {
      weights=new(UseGC) double[nu];
      for(size_t i=0; i < nu; ++i)
        weights[i]=s->weights[i];
    } else weights=NULL;
    
    size_t nuknots=udegree+nu+1;
    uknots=new(UseGC) double[nuknots];
    
    for(size_t i=0; i < nuknots; ++i)
      uknots[i]=s->uknots[i];
    
#ifdef HAVE_LIBGL
    uKnots=new(UseGC) GLfloat[nuknots];
    Controls=new(UseGC)  GLfloat[(weights ? 4 : 3)*nu];
#endif    
  }
  
  bool is3D() {return true;}
  
  void bounds(bbox3& b);
  
  virtual ~drawNurbsPath3() {}

  bool write(prcfile *out, unsigned int *count, vm::array *index,
             vm::array *origin);
  
  void displacement();
  void ratio(pair &b, double (*m)(double, double), bool &first);
    
  void render(GLUnurbs *nurb, double size2,
              const triple& Min, const triple& Max,
              double perspective, bool transparent);
    
  drawElement *transformed(const vm::array& t);
};

}

#endif
