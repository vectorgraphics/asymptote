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
  bool lighton;
  
  bool invisible;
  triple Min,Max;
  static triple c3[16];
  GLfloat *colors;
  bool havecolors;
  
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
              double opacity, double shininess, double PRCshininess,
              double granularity, triple normal, bool lighton,
              const vm::array &pens) :
    straight(straight), opacity(opacity), shininess(shininess),
    PRCshininess(PRCshininess), granularity(granularity), normal(unit(normal)),
    lighton(lighton) {
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
    havecolors=size > 0;
    if(havecolors) {
      colors=new GLfloat[16];
      if(size != 4) reportError(wrongsize);
      storecolor(0,pens,0);
      storecolor(8,pens,1);
      storecolor(12,pens,2);
      storecolor(4,pens,3);
    }
#endif    
  }
  
  drawSurface(const vm::array& t, const drawSurface *s) :
    straight(s->straight), diffuse(s->diffuse), ambient(s->ambient),
    emissive(s->emissive), specular(s->specular), opacity(s->opacity),
    shininess(s->shininess), PRCshininess(s->PRCshininess), 
    granularity(s->granularity), lighton(s->lighton),
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
  
class drawNurb : public drawElement {
protected:
  size_t degreeu,degreev;
  size_t nu,nv;
  Triple *controls;
  double *weights;
  double *knotsu, *knotsv;
  RGBAColour diffuse;
  RGBAColour ambient;
  RGBAColour emissive;
  RGBAColour specular;
  double opacity;
  double shininess;
  double PRCshininess;
  double granularity;
  triple normal;
  bool lighton;
  
  bool invisible;
public:
  drawNurb(size_t degreeu, size_t degreev, size_t nu, size_t nv,
           const vm::array& g, double *knotsu, double *knotsv,
           const vm::array& weight,
           const vm::array&p, double opacity, double shininess,
           double PRCshininess, double granularity) :
    degreeu(degreeu), degreev(degreev), nu(nu), nv(nv),
    knotsu(knotsu), knotsv(knotsv),
    opacity(opacity), shininess(shininess), PRCshininess(PRCshininess),
    granularity(granularity) {

    string wrongsize="Inconsistent nurb parameters";
    if(checkArray(&g) != nu || checkArray(&weight) != nu || checkArray(&p) != 4)
      reportError(wrongsize);

    size_t n=nu*nv;
    controls=new double[n][3];
    weights=new double[n];
    
    size_t k=0;
    for(size_t i=0; i < nu; ++i) {
      vm::array *gi=vm::read<vm::array*>(g,i);
      vm::array *weighti=vm::read<vm::array*>(weight,i);
      if(checkArray(gi) != nv || checkArray(weighti) != nv)  
        reportError(wrongsize);
      for(size_t j=0; j < nv; ++j) {
        triple v=vm::read<triple>(gi,j);
        controls[k][0]=v.getx();
        controls[k][1]=v.gety();
        controls[k][2]=v.getz();
        weights[k]=vm::read<double>(weighti,j);
        ++k;
      }
    }
      
    pen surfacepen=vm::read<camp::pen>(p,0);
    invisible=surfacepen.invisible();
    
    diffuse=rgba(surfacepen);
    ambient=rgba(vm::read<camp::pen>(p,1));
    emissive=rgba(vm::read<camp::pen>(p,2));
    specular=rgba(vm::read<camp::pen>(p,3));
  }
  
  drawNurb(const vm::array& t, const drawNurb *s) :
    degreeu(s->degreeu), degreev(s->degreev), nu(s->nu), nv(s->nv),
    knotsu(s->knotsu), knotsv(s->knotsv),
    diffuse(s->diffuse), ambient(s->ambient),
    emissive(s->emissive), specular(s->specular), opacity(s->opacity),
    shininess(s->shininess), PRCshininess(s->PRCshininess), 
    granularity(s->granularity), invisible(s->invisible) {
    
    size_t n=nu*nv;
    controls=new double[n][3];
    weights=new double[n];
    
    for(size_t i=0; i < n; ++i) {
      const double *c=s->controls[i];
      triple v=run::operator *(t,triple(c[0],c[1],c[2]));
      controls[i][0]=v.getx();
      controls[i][1]=v.gety();
      controls[i][2]=v.getz();
      weights[i]=s->weights[i];
    }
  }
  
  bool is3D() {return true;}
  
  virtual ~drawNurb() {
    delete[] controls;
    delete[] knotsu;
    delete[] knotsv;
    delete[] weights;
  }

  bool write(prcfile *out);
  
  drawElement *transformed(const vm::array& t);
};
  
double norm(double *a, size_t n);
double norm(triple *a, size_t n);

}

#endif
