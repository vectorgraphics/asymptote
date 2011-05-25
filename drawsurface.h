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

#ifdef HAVE_GL
void storecolor(GLfloat *colors, int i, const vm::array &pens, int j);
  
inline void initMatrix(GLfloat *v, double x, double ymin, double zmin,
                       double ymax, double zmax)
{
  v[0]=x;
  v[1]=ymin;
  v[2]=zmin;
  v[3]=1.0;
  v[4]=x;
  v[5]=ymin;
  v[6]=zmax;
  v[7]=1.0;
  v[8]=x;
  v[9]=ymax;
  v[10]=zmin;
  v[11]=1.0;
  v[12]=x;
  v[13]=ymax;
  v[14]=zmax;
  v[15]=1.0;
}
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
  RGBAColour *colors;
  double opacity;
  double shininess;
  double PRCshininess;
  triple normal;
  bool invisible;
  bool lighton;
  Interaction interaction;
  
  triple Min,Max;
  bool prc;
  
#ifdef HAVE_GL
  triple d; // Maximum deviation of surface from a quadrilateral.
  triple dperp;
#endif  
  
public:
  drawSurface(const vm::array& g, triple center, bool straight,
              const vm::array&p, double opacity, double shininess,
              double PRCshininess, triple normal, const vm::array &pens,
              bool lighton, Interaction interaction, bool prc) :
    center(center), straight(straight), opacity(opacity), shininess(shininess),
    PRCshininess(PRCshininess), normal(unit(normal)), lighton(lighton),
    interaction(interaction), prc(prc) {
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
    
    int size=checkArray(&pens);
    if(size > 0) {
      if(size != 4) reportError(wrongsize);
      colors=new(UseGC) RGBAColour[4];
      colors[0]=rgba(vm::read<camp::pen>(pens,0));
      colors[1]=rgba(vm::read<camp::pen>(pens,3));
      colors[2]=rgba(vm::read<camp::pen>(pens,1));
      colors[3]=rgba(vm::read<camp::pen>(pens,2));
    } else colors=NULL;
  }
  
  drawSurface(const vm::array& t, const drawSurface *s) :
    straight(s->straight), diffuse(s->diffuse), ambient(s->ambient),
    emissive(s->emissive), specular(s->specular), opacity(s->opacity),
    shininess(s->shininess), PRCshininess(s->PRCshininess), 
    invisible(s->invisible), lighton(s->lighton),
    interaction(s->interaction), prc(s->prc) { 
    
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
    
#ifdef HAVE_GL
    center=run::operator *(t,s->center);
    normal=run::multshiftless(t,s->normal);
#endif    
    
    if(s->colors) {
      colors=new(UseGC) RGBAColour[4];
      for(size_t i=0; i < 4; ++i)
        colors[i]=s->colors[i];
    } else colors=NULL;
  }
  
  bool is3D() {return true;}
  
  void bounds(bbox3& b);
  
  void ratio(pair &b, double (*m)(double, double), double fuzz, bool &first);
  
  virtual ~drawSurface() {}

  bool write(prcfile *out, unsigned int *, vm::array *, vm::array *, double,
             groupsmap&);
  
  void displacement();
  
#ifdef HAVE_GL
  void initMatrix(GLfloat *v1, GLfloat *v2) {
    camp::initMatrix(v1,Min.getx(),Min.gety(),Min.getz(),Max.gety(),Max.getz());
    camp::initMatrix(v2,Max.getx(),Min.gety(),Min.getz(),Max.gety(),Max.getz());
  }
#endif  

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
  triple normal;
  bool invisible;
  bool lighton;
  
  triple Min,Max;
  
#ifdef HAVE_GL
  GLfloat *colors;
  GLfloat *Controls;
  GLfloat *uKnots;
  GLfloat *vKnots;
#endif  
  
public:
  drawNurbs(const vm::array& g, const vm::array* uknot, const vm::array* vknot,
            const vm::array* weight, const vm::array&p, double opacity,
            double shininess, double PRCshininess, const vm::array &pens,
            bool lighton) : opacity(opacity), shininess(shininess),
            PRCshininess(PRCshininess), lighton(lighton) {
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
    
    run::copyArrayC(uknots,uknot,0,UseGC);
    run::copyArrayC(vknots,vknot,0,UseGC);
    
    pen surfacepen=vm::read<camp::pen>(p,0);
    invisible=surfacepen.invisible();
    
    diffuse=rgba(surfacepen);
    ambient=rgba(vm::read<camp::pen>(p,1));
    emissive=rgba(vm::read<camp::pen>(p,2));
    specular=rgba(vm::read<camp::pen>(p,3));
    
#ifdef HAVE_GL
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
    invisible(s->invisible), lighton(s->lighton) {
    
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
    
#ifdef HAVE_GL
    Controls=NULL;
    if(s->colors) {
      colors=new(UseGC) GLfloat[16];
      for(size_t i=0; i < 16; ++i)
        colors[i]=s->colors[i];
    } else colors=NULL;
#endif    
  }
  
  bool is3D() {return true;}
  
  void bounds(bbox3& b);
  
  virtual ~drawNurbs() {}

  bool write(prcfile *out, unsigned int *, vm::array *, vm::array *, double,
             groupsmap&);
  
  void displacement();
  void ratio(pair &b, double (*m)(double, double), double, bool &first);
  
#ifdef HAVE_GL
  void initMatrix(GLfloat *v1, GLfloat *v2) {
    camp::initMatrix(v1,Min.getx(),Min.gety(),Min.getz(),Max.gety(),Max.getz());
    camp::initMatrix(v2,Max.getx(),Min.gety(),Min.getz(),Max.gety(),Max.getz());
  }
#endif  

  void render(GLUnurbs *nurb, double size2,
              const triple& Min, const triple& Max,
              double perspective, bool transparent);
    
  drawElement *transformed(const vm::array& t);
};
  
template<class T>
void copyArray4x4C(T* dest, const vm::array *a)
{
  size_t n=checkArray(a);
  string fourbyfour="4x4 array of doubles expected";
  if(n != 4) reportError(fourbyfour);
  
  for(size_t i=0; i < 4; i++) {
    vm::array *ai=vm::read<vm::array*>(a,i);
    size_t aisize=checkArray(ai);
    if(aisize == 4) {
      T *desti=dest+4*i;
      for(size_t j=0; j < 4; j++) 
        desti[j]=vm::read<T>(ai,j);
    } else reportError(fourbyfour);
  }
}

// Draw a transformed PRC object.
class drawPRC : public drawElement {
protected:
  double T[16];
  RGBAColour diffuse;
  RGBAColour ambient;
  RGBAColour emissive;
  RGBAColour specular;
  double opacity;
  double shininess;
  bool invisible;
public:
  drawPRC(const vm::array& t, const vm::array&p, double opacity,
             double shininess) : 
    opacity(opacity), shininess(shininess) {
    
    copyArray4x4C<double>(T,&t);

    string needfourpens="array of 4 pens required";
    if(checkArray(&p) != 4)
      reportError(needfourpens);
    
    pen surfacepen=vm::read<camp::pen>(p,0);
    invisible=surfacepen.invisible();
    
    diffuse=rgba(surfacepen);
    ambient=rgba(vm::read<camp::pen>(p,1));
    emissive=rgba(vm::read<camp::pen>(p,2));
    specular=rgba(vm::read<camp::pen>(p,3));
  }
  
  drawPRC(const vm::array& t, const drawPRC *s) :
    diffuse(s->diffuse), ambient(s->ambient), emissive(s->emissive),
    specular(s->specular), opacity(s->opacity),
    shininess(s->shininess), invisible(s->invisible) {
    
    double S[16];
    copyArray4x4C<double>(S,&t);
    
    const double *R=s->T;
    for(unsigned i=0; i < 16; i += 4) {
      double s0=S[i+0];
      double s1=S[i+1];
      double s2=S[i+2];
      double s3=S[i+3];
      T[i]=s0*R[0]+s1*R[4]+s2*R[8]+s3*R[12];
      T[i+1]=s0*R[1]+s1*R[5]+s2*R[9]+s3*R[13];
      T[i+2]=s0*R[2]+s1*R[6]+s2*R[10]+s3*R[14];
      T[i+3]=s0*R[3]+s1*R[7]+s2*R[11]+s3*R[15];
    }
  }
  
  bool write(prcfile *out, unsigned int *, vm::array *, vm::array *, double,
             groupsmap&) {
    return true;
  }
};
  
// Draw a PRC unit sphere.
class drawSphere : public drawPRC {
  bool half;
  int type;
public:
  drawSphere(const vm::array& t, bool half, const vm::array&p, double opacity,
             double shininess, int type) :
    drawPRC(t,p,opacity,shininess), half(half), type(type) {}

  drawSphere(const vm::array& t, const drawSphere *s) :
    drawPRC(t,s), half(s->half), type(s->type) {}
    
  void P(Triple& t, double x, double y, double z);
  
  bool write(prcfile *out, unsigned int *, vm::array *, vm::array *, double,
             groupsmap&);
  
  drawElement *transformed(const vm::array& t) {
      return new drawSphere(t,this);
  }
};
  
// Draw a PRC unit cylinder.
class drawCylinder : public drawPRC {
public:
  drawCylinder(const vm::array& t, const vm::array&p, double opacity,
             double shininess) :
    drawPRC(t,p,opacity,shininess) {}

  drawCylinder(const vm::array& t, const drawCylinder *s) :
    drawPRC(t,s) {}
    
  bool write(prcfile *out, unsigned int *, vm::array *, vm::array *, double,
             groupsmap&);
  
  drawElement *transformed(const vm::array& t) {
      return new drawCylinder(t,this);
  }
};
  
// Draw a PRC unit disk.
class drawDisk : public drawPRC {
public:
  drawDisk(const vm::array& t, const vm::array&p, double opacity,
             double shininess) :
    drawPRC(t,p,opacity,shininess) {}

  drawDisk(const vm::array& t, const drawDisk *s) :
    drawPRC(t,s) {}
    
  bool write(prcfile *out, unsigned int *, vm::array *, vm::array *, double,
             groupsmap&);
  
  drawElement *transformed(const vm::array& t) {
      return new drawDisk(t,this);
  }
};
  
// Draw a PRC tube.
class drawTube : public drawElement {
protected:
  path3 center;
  path3 g;
  RGBAColour diffuse;
  RGBAColour ambient;
  RGBAColour emissive;
  RGBAColour specular;
  double opacity;
  double shininess;
  bool invisible;
public:
  drawTube(path3 center, path3 g, const vm::array&p, double opacity,
             double shininess) : 
    center(center), g(g), opacity(opacity), shininess(shininess) {
    string needfourpens="array of 4 pens required";
    if(checkArray(&p) != 4)
      reportError(needfourpens);
    
    pen surfacepen=vm::read<camp::pen>(p,0);
    invisible=surfacepen.invisible();
    
    diffuse=rgba(surfacepen);
    ambient=rgba(vm::read<camp::pen>(p,1));
    emissive=rgba(vm::read<camp::pen>(p,2));
    specular=rgba(vm::read<camp::pen>(p,3));
  }
  
  drawTube(const vm::array& t, const drawTube *s) :
    center(camp::transformed(t,s->center)), g(camp::transformed(t,s->g)), 
    diffuse(s->diffuse), ambient(s->ambient), emissive(s->emissive),
    specular(s->specular), opacity(s->opacity),
    shininess(s->shininess), invisible(s->invisible) {
  }
  
  bool write(prcfile *out, unsigned int *, vm::array *, vm::array *, double,
             groupsmap&);
                        
  drawElement *transformed(const vm::array& t) {
      return new drawTube(t,this);
  }
};

}

#endif
