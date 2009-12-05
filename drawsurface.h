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

enum Interaction {EMBEDDED=0,BILLBOARD};

#ifdef HAVE_LIBGL
void storecolor(GLfloat *colors, int i, const vm::array &pens, int j);
#endif  
  
class drawSurface : public drawElement {
protected:
  Triple *controls;
  Triple vertices[4];
  triple center;
  bool straight;
  RGBAColour diffuse;
  RGBAColour ambient;
  RGBAColour emissive;
  RGBAColour specular;
  double opacity;
  double shininess;
  double PRCshininess;
  double granularity;
  triple normal;
  bool invisible;
  bool lighton;
  string name;
  Interaction interaction;
  
  triple Min,Max;
  
#ifdef HAVE_LIBGL
  GLfloat *colors;
  triple d; // Maximum deviation of surface from a quadrilateral.
  triple dperp;
#endif  
  
public:
  static const triple zero;

  drawSurface(const vm::array& g, triple center, bool straight,
              const vm::array&p, double opacity, double shininess,
              double PRCshininess, double granularity, triple normal,
              const vm::array &pens, bool lighton, const string& name,
              Int interaction) :
    center(center), straight(straight), opacity(opacity), shininess(shininess),
    PRCshininess(PRCshininess), granularity(granularity), normal(unit(normal)),
    lighton(lighton), name(name), interaction((Interaction) interaction) {
    string wrongsize=
      "Bezier surface patch requires 4x4 array of triples and array of 4 pens";
    if(checkArray(&g) != 4 || checkArray(&p) != 4)
      reportError(wrongsize);
    
    bool havenormal=normal != zero;
  
    vm::array *g0=vm::read<vm::array*>(g,0);
    vm::array *g3=vm::read<vm::array*>(g,3);
    if(checkArray(g0) != 4 || checkArray(g3) != 4)
      reportError(wrongsize);
    store(vertices[0],vm::read<triple>(g0,0));
    store(vertices[1],vm::read<triple>(g0,3));
    store(vertices[2],vm::read<triple>(g3,0));
    store(vertices[3],vm::read<triple>(g3,3));
    
    if(!havenormal || !straight) {
      size_t k=0;
      controls=new(UseGC) Triple[16];
      for(size_t i=0; i < 4; ++i) {
        vm::array *gi=vm::read<vm::array*>(g,i);
        if(checkArray(gi) != 4) 
          reportError(wrongsize);
        for(size_t j=0; j < 4; ++j)
          store(controls[k++],vm::read<triple>(gi,j));
      }
    } else controls=NULL;
    
    pen surfacepen=vm::read<camp::pen>(p,0);
    invisible=surfacepen.invisible();
    
    diffuse=rgba(surfacepen);
    ambient=rgba(vm::read<camp::pen>(p,1));
    emissive=rgba(vm::read<camp::pen>(p,2));
    specular=rgba(vm::read<camp::pen>(p,3));
    
#ifdef HAVE_LIBGL
    int size=checkArray(&pens);
    if(size > 0) {
      if(size != 4) reportError(wrongsize);
      colors=new(UseGC) GLfloat[16];
      storecolor(colors,0,pens,0);
      storecolor(colors,8,pens,1);
      storecolor(colors,12,pens,2);
      storecolor(colors,4,pens,3);
    } else colors=NULL;
#endif    
  }
  
  drawSurface(const vm::array& t, const drawSurface *s) :
    straight(s->straight), diffuse(s->diffuse), ambient(s->ambient),
    emissive(s->emissive), specular(s->specular), opacity(s->opacity),
    shininess(s->shininess), PRCshininess(s->PRCshininess), 
    granularity(s->granularity), invisible(s->invisible),
    lighton(s->lighton), name(s->name), interaction(s->interaction) {
    
    for(size_t i=0; i < 4; ++i) {
      const double *c=s->vertices[i];
      store(vertices[i],run::operator *(t,triple(c[0],c[1],c[2])));
    }
    
    if(s->controls) {
      controls=new(UseGC) Triple[16];
      for(size_t i=0; i < 16; ++i) {
        const double *c=s->controls[i];
        store(controls[i],run::operator *(t,triple(c[0],c[1],c[2])));
      }
    } else controls=NULL;
    
    center=run::operator *(t,s->center);
    normal=run::multshiftless(t,s->normal);
    
#ifdef HAVE_LIBGL
    if(s->colors) {
      colors=new(UseGC) GLfloat[16];
      for(int i=0; i < 16; ++i)
        colors[i]=s->colors[i];
    } else colors=NULL;
#endif    
  }
  
  bool is3D() {return true;}
  
  void bounds(bbox3& b);
  
  void ratio(pair &b, double (*m)(double, double), double fuzz, bool &first);
  
  virtual ~drawSurface() {}

  bool write(prcfile *out, unsigned int *count, vm::array *index,
             vm::array *origin);
  
  void displacement();
  void render(GLUnurbs *nurb, double, const triple& Min, const triple& Max,
              double perspective, bool transparent);
  
  drawElement *transformed(const vm::array& t);
};
  
class drawNurbs : public drawElement {
protected:
  size_t udegree,vdegree;
  size_t nu,nv;
  Triple *controls;
  double *weights;
  double *uknots, *vknots;
  RGBAColour diffuse;
  RGBAColour ambient;
  RGBAColour emissive;
  RGBAColour specular;
  double opacity;
  double shininess;
  double PRCshininess;
  double granularity;
  triple normal;
  bool invisible;
  bool lighton;
  string name;
  
  triple Min,Max;
  
#ifdef HAVE_LIBGL
  GLfloat *colors;
  GLfloat *Controls;
  GLfloat *uKnots;
  GLfloat *vKnots;
#endif  
  
public:
  drawNurbs(const vm::array& g, const vm::array* uknot, const vm::array* vknot,
            const vm::array* weight, const vm::array&p, double opacity,
            double shininess, double PRCshininess, double granularity,
            const vm::array &pens, bool lighton, const string& name) :
    opacity(opacity), shininess(shininess), PRCshininess(PRCshininess),
    granularity(granularity), lighton(lighton), name(name) {
    size_t weightsize=checkArray(weight);
    
    string wrongsize="Inconsistent NURBS data";
    nu=checkArray(&g);
    
    if(nu == 0 || (weightsize != 0 && weightsize != nu) || checkArray(&p) != 4)
      reportError(wrongsize);
    
    vm::array *g0=vm::read<vm::array*>(g,0);
    nv=checkArray(g0);
    
    size_t n=nu*nv;
    controls=new(UseGC) Triple[n];
    
    size_t k=0;
    for(size_t i=0; i < nu; ++i) {
      vm::array *gi=vm::read<vm::array*>(g,i);
      if(checkArray(gi) != nv)  
        reportError(wrongsize);
      for(size_t j=0; j < nv; ++j)
        store(controls[k++],vm::read<triple>(gi,j));
    }
      
    if(weightsize > 0) {
      size_t k=0;
      weights=new(UseGC) double[n];
      for(size_t i=0; i < nu; ++i) {
        vm::array *weighti=vm::read<vm::array*>(weight,i);
        if(checkArray(weighti) != nv)  
          reportError(wrongsize);
        for(size_t j=0; j < nv; ++j)
          weights[k++]=vm::read<double>(weighti,j);
      }
    } else weights=NULL;
      
    size_t nuknots=checkArray(uknot);
    size_t nvknots=checkArray(vknot);
    
    if(nuknots <= nu+1 || nuknots > 2*nu || nvknots <= nv+1 || nvknots > 2*nv)
      reportError(wrongsize);

    udegree=nuknots-nu-1;
    vdegree=nvknots-nv-1;
    
    uknots=run::copyArrayC(uknot,0,NoGC);
    vknots=run::copyArrayC(vknot,0,NoGC);
    
    pen surfacepen=vm::read<camp::pen>(p,0);
    invisible=surfacepen.invisible();
    
    diffuse=rgba(surfacepen);
    ambient=rgba(vm::read<camp::pen>(p,1));
    emissive=rgba(vm::read<camp::pen>(p,2));
    specular=rgba(vm::read<camp::pen>(p,3));
    
#ifdef HAVE_LIBGL
    Controls=NULL;
    int size=checkArray(&pens);
    if(size > 0) {
      colors=new(UseGC) GLfloat[16];
      if(size != 4) reportError(wrongsize);
      storecolor(colors,0,pens,0);
      storecolor(colors,8,pens,1);
      storecolor(colors,12,pens,2);
      storecolor(colors,4,pens,3);
    } else colors=NULL;
#endif  
  }
  
  drawNurbs(const vm::array& t, const drawNurbs *s) :
    udegree(s->udegree), vdegree(s->vdegree), nu(s->nu), nv(s->nv),
    diffuse(s->diffuse), ambient(s->ambient),
    emissive(s->emissive), specular(s->specular), opacity(s->opacity),
    shininess(s->shininess), PRCshininess(s->PRCshininess), 
    granularity(s->granularity), invisible(s->invisible), lighton(s->lighton),
    name(s->name) {
    
    size_t n=nu*nv;
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
    
    size_t nuknots=udegree+nu+1;
    size_t nvknots=vdegree+nv+1;
    uknots=new(UseGC) double[nuknots];
    vknots=new(UseGC) double[nvknots];
    
    for(size_t i=0; i < nuknots; ++i)
      uknots[i]=s->uknots[i];
    
    for(size_t i=0; i < nvknots; ++i)
      vknots[i]=s->vknots[i];
    
#ifdef HAVE_LIBGL
    Controls=NULL;
    if(s->colors) {
      colors=new(UseGC) GLfloat[16];
      for(int i=0; i < 16; ++i)
        colors[i]=s->colors[i];
    } else colors=NULL;
#endif    
  }
  
  bool is3D() {return true;}
  
  void bounds(bbox3& b);
  
  virtual ~drawNurbs() {}

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
