/*****
 * drawelement.h
 * Andy Hammerlindl 2002/06/06
 *
 * Abstract base class of any drawable item in camp.
 *****/

#ifndef DRAWELEMENT_H
#define DRAWELEMENT_H

#include <vector>

#include "common.h"
#include "bbox.h"
#include "bbox3.h"
#include "pen.h"
#include "psfile.h"
#include "texfile.h"
#include "prcfile.h"
#include "glrender.h"
#include "arrayop.h"

namespace camp {

enum Interaction {EMBEDDED=0,BILLBOARD};

class box {
  pair p[4];
public:
  
  box() {}
  box(const pair& a, const pair& b, const pair& c, const pair& d) {
    p[0]=a; p[1]=b; p[2]=c; p[3]=d;
  }
  
// Returns true if the line a--b intersects box b.
  bool intersect(const pair& a, const pair& b) const
  {
    for(Int i=0; i < 4; ++i) {
      pair A=p[i];
      pair B=p[i < 3 ? i+1 : 0];
      double de=(b.x-a.x)*(A.y-B.y)-(A.x-B.x)*(b.y-a.y);
      if(de != 0.0) {
        de=1.0/de;
        double t=((A.x-a.x)*(A.y-B.y)-(A.x-B.x)*(A.y-a.y))*de;
        double T=((b.x-a.x)*(A.y-a.y)-(A.x-a.x)*(b.y-a.y))*de;
        if(0 <= t && t <= 1 && 0 <= T && T <= 1) return true;
      }
    }
    return false;
  }
  
  pair operator [] (Int i) const {return p[i];}
  
  bool intersect(const box& b) const {
    for(Int i=0; i < 4; ++i) {
      pair A=b[i];
      pair B=b[i < 3 ? i+1 : 0];
      if(intersect(A,B)) return true;
    }
    return false;
  }
  
// Returns true iff the point z lies in the region bounded by b.
  bool inside(const pair& z) const {
    bool c=false;
    for(Int i=0; i < 3; ++i) {
      pair pi=p[i];
      pair pj=p[i < 3 ? i+1 : 0];
      if(((pi.y <= z.y && z.y < pj.y) || (pj.y <= z.y && z.y < pi.y)) &&
         z.x < pi.x+(pj.x-pi.x)*(z.y-pi.y)/(pj.y-pi.y)) c=!c;
    }
    return c;
  }
  
  double xmax() {
    return max(max(max(p[0].x,p[1].x),p[2].x),p[3].x);
  }
  
  double ymax() {
    return max(max(max(p[0].y,p[1].y),p[2].y),p[3].y);
  }
  
  double xmin() {
    return min(min(min(p[0].x,p[1].x),p[2].x),p[3].x);
  }
  
  double ymin() {
    return min(min(min(p[0].y,p[1].y),p[2].y),p[3].y);
  }
  
};
  
typedef mem::vector<box> boxvector;
  
typedef mem::list<bbox> bboxlist;
  
typedef mem::map<CONST string,unsigned> groupmap;
typedef mem::vector<groupmap> groupsmap;

class drawElement : public gc
{
public:
  virtual ~drawElement() {}
  
  static pen lastpen;  
  static const triple zero;
  
  // Adjust the bbox of the picture based on the addition of this
  // element. The iopipestream is needed for determining label sizes.
  virtual void bounds(bbox&, iopipestream&, boxvector&, bboxlist&) {}
  virtual void bounds(bbox3&) {}

  // Compute bounds on ratio (x,y)/z for 3d picture (not cached).
  virtual void ratio(pair &b, double (*m)(double, double), double fuzz,
                     bool &first){}

  virtual bool islabel() {return false;}

  virtual bool islayer() {return false;}

  virtual bool is3D() {return false;}

// Implement element as raw SVG code?
  virtual bool svg() {return false;}
  
// Implement SVG element as png image?
  virtual bool svgpng() {return false;}
  
  virtual bool beginclip() {return false;}
  virtual bool endclip() {return false;}
  
  virtual bool begingroup() {return false;}

  virtual bool endgroup() {return false;}

  virtual void save(bool b) {}
  
  // Output to a PostScript file
  virtual bool draw(psfile *) {
    return false;
  }

  // Output to a TeX file
  virtual bool write(texfile *, const bbox&) {
    return false;
  }

  // Output to a PRC file
  // The array origin contains the points about which to rotate billboard labels
  virtual bool write(prcfile *out, unsigned int *count, vm::array *index,
                     vm::array *origin, double compressionlimit,
                     groupsmap& groups) {
    return false;
  }

  // Used to compute deviation of a surface from a quadrilateral.
  virtual void displacement() {}

  // Render with OpenGL
  virtual void render(GLUnurbs *nurb, double size2, 
                      const triple& Min, const triple& Max,
                      double perspective, bool transparent) {}

  // Transform as part of a picture.
  virtual drawElement *transformed(const transform&) {
    return this;
  }
  
  virtual drawElement *transformed(const vm::array&) {
    return this;
  }
};

// Base class for drawElements that involve paths.
class drawPathBase : public virtual drawElement {
protected:
  path p;

  path transpath(const transform& t) const {
    return p.transformed(t);
  }

public:
  drawPathBase() {}
  drawPathBase(path p) : p(p) {}

  virtual ~drawPathBase() {}

  virtual void bounds(bbox& b, iopipestream&, boxvector&, bboxlist&) {
    b += p.bounds();
  }
  
  virtual void writepath(psfile *out,bool) {
    out->write(p);
  }
  
  virtual void writeclippath(psfile *out, bool newpath=true) {
    out->writeclip(p,newpath);
  }
  
  virtual void writeshiftedpath(texfile *out) {
    out->writeshifted(p);
  }
  
};

// Base class for drawElements that involve paths and pens.
class drawPathPenBase : public drawPathBase {
protected:
  pen pentype;

  pen transpen(const transform& t) const {
    return camp::transformed(shiftless(t),pentype);
  }

public:
  drawPathPenBase(path p, pen pentype) : 
    drawPathBase(p), pentype(pentype) {}
  
  drawPathPenBase(pen pentype) :
    pentype(pentype) {}
  
  virtual bool empty() {
    return p.empty();
  }
  
  virtual bool cyclic() {
    return p.cyclic();
  }
  
  void strokebounds(bbox& b, const path& p);
    
  virtual void penSave(psfile *out)
  {
    if (!pentype.getTransform().isIdentity())
      out->gsave();
  }
  
  virtual void penTranslate(psfile *out)
  {
    out->translate(shiftpair(pentype.getTransform()));
  }

  virtual void penConcat(psfile *out)
  {
    out->concat(shiftless(pentype.getTransform()));
  }

  virtual void penRestore(psfile *out)
  {
    if (!pentype.getTransform().isIdentity())
      out->grestore();
  }
  
  
};
  
// Base class for drawElements that involve superpaths and pens.
class drawSuperPathPenBase : public drawPathPenBase {
protected:
  vm::array P;
  size_t size;
  bbox bpath;

  vm::array transpath(const transform& t) const {
    vm::array *Pt=new vm::array(size);
    for(size_t i=0; i < size; i++)
      (*Pt)[i]=vm::read<path>(P,i).transformed(t);
    return *Pt;
  }
  
public:
  drawSuperPathPenBase(const vm::array& P, pen pentype) :
    drawPathPenBase(pentype), P(P), size(P.size()) {}

  bool empty() {
    for(size_t i=0; i < size; i++) 
      if(vm::read<path>(P,i).size() != 0) return false;
    return true;
  }
  
  bool cyclic() {
    for(size_t i=0; i < size; i++) 
      if(!vm::read<path>(P,i).cyclic()) return false;
    return true;
  }
  
  void bounds(bbox& b, iopipestream&, boxvector&, bboxlist&) {
    for(size_t i=0; i < size; i++)
      bpath += vm::read<path>(P,i).bounds();
    b += bpath;
  }
  
  void strokepath(psfile *out) {
    // strokepath and evenodd are incompatible
    static pen zerowinding=pen((FillRule) ZEROWINDING);
    pentype=pentype+zerowinding;
    out->setpen(pentype);
    out->strokepath();
  }
  
  void strokebounds(bbox& b) {
    for(size_t i=0; i < size; i++)
      drawPathPenBase::strokebounds(bpath,vm::read<path>(P,i));
    b += bpath;
  }
  
  void writepath(psfile *out, bool newpath=true) {
    if(size > 0) out->write(vm::read<path>(P,0),newpath);
    for(size_t i=1; i < size; i++)
      out->write(vm::read<path>(P,i),false);
  }
  
  void writeclippath(psfile *out, bool newpath=true) {
    if(size > 0) out->writeclip(vm::read<path>(P,0),newpath);
    for(size_t i=1; i < size; i++)
      out->writeclip(vm::read<path>(P,i),false);
  }
  
  void writeshiftedpath(texfile *out) {
    for(size_t i=0; i < size; i++) 
      out->writeshifted(vm::read<path>(P,i),i == 0);
  }
};
 
}

GC_DECLARE_PTRFREE(camp::box);
GC_DECLARE_PTRFREE(camp::drawElement);

#endif

