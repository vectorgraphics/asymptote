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

namespace run {
extern double *copyArrayC(const array *a, size_t dim=0);
}

namespace camp {

void storecolor(GLfloat *colors, int i, const vm::array &pens, int j);
  
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
  double PRCshininess;
  double granularity;
  triple normal;
  bool invisible;
  bool havecolors;
  bool lighton;
  
  triple Min,Max;
  static triple c3[16];
  GLfloat *colors;
  
#ifdef HAVE_LIBGL
  GLfloat c[48];
  triple d; // Maximum deviation of surface from a quadrilateral.
  triple dperp;
  GLfloat v1[16];
  GLfloat v2[16];
  GLfloat Normal[3];
  bool havenormal;
  bool havetransparency;
#endif  
  
public:
  drawSurface(const vm::array& g, bool straight, const vm::array&p,
              double opacity, double shininess, double PRCshininess,
              double granularity, triple normal, const vm::array &pens,
              bool lighton) :
    straight(straight), opacity(opacity), shininess(shininess),
    PRCshininess(PRCshininess), granularity(granularity), normal(unit(normal)),
    havecolors(false), lighton(lighton) {
    string wrongsize=
      "Bezier surface patch requires 4x4 array of triples and array of 4 pens";
    if(checkArray(&g) != 4 || checkArray(&p) != 4)
      reportError(wrongsize);
    
    size_t k=0;
    for(size_t i=0; i < 4; ++i) {
      vm::array *gi=vm::read<vm::array*>(g,i);
      if(checkArray(gi) != 4) 
        reportError(wrongsize);
      for(size_t j=0; j < 4; ++j) {
        triple v=vm::read<triple>(gi,j);
        controls[k][0]=v.getx();
        controls[k][1]=v.gety();
        controls[k][2]=v.getz();
        ++k;
      }
    }
    
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
      havecolors=true;
      colors=new GLfloat[16];
      storecolor(colors,0,pens,0);
      storecolor(colors,8,pens,1);
      storecolor(colors,12,pens,2);
      storecolor(colors,4,pens,3);
    }
#endif    
  }
  
  drawSurface(const vm::array& t, const drawSurface *s) :
    straight(s->straight), diffuse(s->diffuse), ambient(s->ambient),
    emissive(s->emissive), specular(s->specular), opacity(s->opacity),
    shininess(s->shininess), PRCshininess(s->PRCshininess), 
    granularity(s->granularity), invisible(s->invisible),
    havecolors(s->havecolors), lighton(s->lighton)
  {
    for(size_t i=0; i < 16; ++i) {
      const double *c=s->controls[i];
      triple v=run::operator *(t,triple(c[0],c[1],c[2]));
      controls[i][0]=v.getx();
      controls[i][1]=v.gety();
      controls[i][2]=v.getz();
    }
    normal=run::multshiftless(t,s->normal);
    
    if(havecolors) {
      colors=new GLfloat[16];
      for(int i=0; i < 16; ++i)
        colors[i]=s->colors[i];
    }
  }
  
  bool is3D() {return true;}
  
  void bounds(bbox3& b);
  
  void ratio(pair &b, double (*m)(double, double), bool &first);
  
  virtual ~drawSurface() {
    if(havecolors)
      delete[] colors;
  }

  bool write(prcfile *out);
  
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
  bool havecolors;
  bool lighton;
  
  triple Min,Max;
  GLfloat *colors;
#ifdef HAVE_LIBGL
  GLfloat *c;
  GLfloat v1[16];
  GLfloat v2[16];
  GLfloat *uKnots;
  GLfloat *vKnots;
  bool havetransparency;
#endif  
  
public:
  drawNurbs(const vm::array& g, const vm::array* uknot, const vm::array* vknot,
           const vm::array* weight, const vm::array&p, double opacity,
            double shininess, double PRCshininess, double granularity,
            const vm::array &pens, bool lighton) :
    opacity(opacity), shininess(shininess), PRCshininess(PRCshininess),
    granularity(granularity), havecolors(false), lighton(lighton) {
    size_t weightsize=checkArray(weight);
    
    string wrongsize="Inconsistent NURBS data";
    nu=checkArray(&g);
    
    if(nu == 0 || (weightsize != 0 && weightsize != nu) || checkArray(&p) != 4)
      reportError(wrongsize);
    
    vm::array *g0=vm::read<vm::array*>(g,0);
    nv=checkArray(g0);
    
    size_t n=nu*nv;
    controls=new double[n][3];
    
    size_t k=0;
    for(size_t i=0; i < nu; ++i) {
      vm::array *gi=vm::read<vm::array*>(g,i);
      if(checkArray(gi) != nv)  
        reportError(wrongsize);
      for(size_t j=0; j < nv; ++j) {
        triple v=vm::read<triple>(gi,j);
        controls[k][0]=v.getx();
        controls[k][1]=v.gety();
        controls[k][2]=v.getz();
        ++k;
      }
    }
      
    if(weightsize == 0)
      weights=NULL;
    else {
      size_t k=0;
      weights=new double[n];
      for(size_t i=0; i < nu; ++i) {
        vm::array *weighti=vm::read<vm::array*>(weight,i);
        if(checkArray(weighti) != nv)  
          reportError(wrongsize);
        for(size_t j=0; j < nv; ++j) {
          weights[k]=vm::read<double>(weighti,j);
          ++k;
        }
      }
    }
      
    size_t nuknots=checkArray(uknot);
    size_t nvknots=checkArray(vknot);
    
    if(nuknots <= nu+1 || nuknots > 2*nu || nvknots <= nv+1 || nvknots > 2*nv)
      reportError(wrongsize);

    udegree=nuknots-nu-1;
    vdegree=nvknots-nv-1;
    
    uknots=run::copyArrayC(uknot);
    vknots=run::copyArrayC(vknot);
    
    pen surfacepen=vm::read<camp::pen>(p,0);
    invisible=surfacepen.invisible();
    
    diffuse=rgba(surfacepen);
    ambient=rgba(vm::read<camp::pen>(p,1));
    emissive=rgba(vm::read<camp::pen>(p,2));
    specular=rgba(vm::read<camp::pen>(p,3));
#ifdef HAVE_LIBGL
    int size=checkArray(&pens);
    havecolors=size > 0;
    if(havecolors) {
      colors=new GLfloat[16];
      if(size != 4) reportError(wrongsize);
      storecolor(colors,0,pens,0);
      storecolor(colors,8,pens,1);
      storecolor(colors,12,pens,2);
      storecolor(colors,4,pens,3);
    }
    
    uKnots=new GLfloat[nuknots];
    vKnots=new GLfloat[nvknots];
    c=new GLfloat[4*n];
#endif  
  }
  
  drawNurbs(const vm::array& t, const drawNurbs *s) :
    udegree(s->udegree), vdegree(s->vdegree), nu(s->nu), nv(s->nv),
    diffuse(s->diffuse), ambient(s->ambient),
    emissive(s->emissive), specular(s->specular), opacity(s->opacity),
    shininess(s->shininess), PRCshininess(s->PRCshininess), 
    granularity(s->granularity), invisible(s->invisible),
    havecolors(s->havecolors), lighton(s->lighton) {
    
    size_t n=nu*nv;
    controls=new double[n][3];
      
    for(size_t i=0; i < n; ++i) {
      const double *c=s->controls[i];
      triple v=run::operator *(t,triple(c[0],c[1],c[2]));
      controls[i][0]=v.getx();
      controls[i][1]=v.gety();
      controls[i][2]=v.getz();
    }
    
    if(s->weights == NULL) 
      weights=NULL;
    else {
      weights=new double[n];
      for(size_t i=0; i < n; ++i)
        weights[i]=s->weights[i];
    }
    
    size_t nuknots=udegree+nu+1;
    size_t nvknots=vdegree+nv+1;
    uknots=new double[nuknots];
    vknots=new double[nvknots];
    
    for(size_t i=0; i < nuknots; ++i)
      uknots[i]=s->uknots[i];
    
    for(size_t i=0; i < nvknots; ++i)
      vknots[i]=s->vknots[i];
    
#ifdef HAVE_LIBGL
    uKnots=new GLfloat[nuknots];
    vKnots=new GLfloat[nvknots];
    c=new GLfloat[(weights == NULL ? 3 : 4)*n];
    
    if(havecolors) {
      colors=new GLfloat[16];
      for(int i=0; i < 16; ++i)
        colors[i]=s->colors[i];
    }
#endif    
  }
  
  bool is3D() {return true;}
  
  void bounds(bbox3& b);
  
  virtual ~drawNurbs() {
    delete[] c;
    delete[] vknots;
    delete[] uknots;
    if(weights != NULL) 
      delete[] weights;
#ifdef HAVE_LIBGL
    delete[] controls;
    delete[] vKnots;
    delete[] uKnots;
    if(havecolors)
      delete[] colors; 
#endif    
  }

  bool write(prcfile *out);
  
  void displacement();
  void ratio(pair &b, double (*m)(double, double), bool &first);
    
  void render(GLUnurbs *nurb, double size2,
              const triple& Min, const triple& Max,
              double perspective, bool transparent);
    
  drawElement *transformed(const vm::array& t);
};
  
double norm(double *a, size_t n);
double norm(triple *a, size_t n);

}

#endif
