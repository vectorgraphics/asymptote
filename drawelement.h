/*****
 * drawelement.h
 * Andy Hammerlindl 2002/06/06
 *
 * Abstract base class of any drawable item in camp.
 *****/

#ifndef DRAWELEMENT_H
#define DRAWELEMENT_H

#include <list>
#include <string>
#include <vector>

#include "bbox.h"
#include "pen.h"
#include "psfile.h"
#include "texfile.h"
#include "pipestream.h"

using std::string;

namespace camp {

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
    for(int i=0; i < 4; ++i) {
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
  
  pair operator [] (int i) const {return p[i];}
  
  bool intersect(const box& b) const {
    for(int i=0; i < 4; ++i) {
      pair A=b[i];
      pair B=b[i < 3 ? i+1 : 0];
      if(intersect(A,B)) return true;
    }
    return false;
  }
  
// Returns true iff the point z lies in the region bounded by b.
  bool inside(const pair& z) const {
    bool c=false;
    for(int i=0; i < 3; ++i) {
      pair pi=p[i];
      pair pj=p[i < 3 ? i+1 : 0];
      if(((pi.y <= z.y && z.y < pj.y) || (pj.y <= z.y && z.y < pi.y)) &&
	 z.x < pi.x+(pj.x-pi.x)*(z.y-pi.y)/(pj.y-pi.y)) c=!c;
    }
    return c;
  }
  
  bool overlap(const box& b) const {
    if(intersect(b)) return true;
    if(inside(b[0])) return true;
    if(b.inside(p[0])) return true;
    return false;
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
  
typedef std::vector<box,gc_allocator<box> > boxvector;
typedef std::list<bbox,gc_allocator<bbox> > bboxlist;
  
class drawElement : public gc
{
public:
  static pen lastpen;  
  
  // Adjust the bbox of the picture based on the addition of this
  // element. The iopipestream is needed for determining label sizes.
  virtual void bounds(bbox&, iopipestream&, boxvector&, bboxlist&) {}

  virtual bool islabel() {return false;}

  virtual bool islayer() {return false;}

  virtual bool begingroup() {return false;}

  virtual bool endgroup() {return false;}

  // Handles its output in a PostScript file
  virtual bool draw(psfile *) {
    return true;
  }

  // Handles its output in a TeX file
  virtual bool write(texfile *) {
    return true;
  }

  // Transform as part of a picture.
  virtual drawElement *transformed(const transform&) {
    return this;
  }
};

// Base class for drawElements that involve paths.
class drawPathBase : public drawElement {
protected:
  path p;

  path transpath(const transform& t) const {
    return p.transformed(t);
  }

public:
  drawPathBase() {}
  drawPathBase(path p) : p(p) {}

  virtual ~drawPathBase() {}

  void bounds(bbox& b, iopipestream&, boxvector&, bboxlist&) {
    b += p.bounds();
  }
};

// Base class for drawElements that involve paths and pens.
class drawPathPenBase : public drawPathBase {
protected:
  vm::array *P;
  pen pentype;
  size_t size;

  // The pen's transform;
  const transform *t()
  {
    return pentype.getTransform();
  }

  pen transpen(const transform& t) const {
    return camp::transformed(new transform(shiftless(t)),pentype);
  }

  vm::array *transPath(const transform& t) const {
    size_t size=(size_t) P->size();
    vm::array *Pt=new vm::array(size);
    for(size_t i=0; i < size; i++)
      (*Pt)[i]=vm::read<path>(P,i).transformed(t);
    return Pt;
  }
  
public:
  drawPathPenBase(path p, pen pentype) : 
    drawPathBase(p), P(NULL), pentype(pentype) {}
  
  drawPathPenBase(vm::array *P, pen pentype) :
    drawPathBase(), P(P), pentype(pentype), size(P->size()) {}

  bool empty() {
    if(P) {
      for(size_t i=0; i < size; i++) 
	if(vm::read<path>(P,i).size() != 0) return false;
      return true;
    } else return p.empty();
  }
  
  bool cyclic() {
    if(P) {
      for(size_t i=0; i < size; i++) 
	if(!vm::read<path>(P,i).cyclic()) return false;
      return true;
    } else return p.cyclic();
  }
  
  void bounds(bbox& b, iopipestream&, boxvector&, bboxlist&) {
    if(P) {
      for(size_t i=0; i < size; i++)
	b += vm::read<path>(P,i).bounds();
    } else b += p.bounds();
  }
  
  void writepath(psfile *out) {
    if(P) {
      for(size_t i=0; i < size; i++) 
	out->write(vm::read<path>(P,i),i == 0);
    } else out->write(p);
  }
  
  virtual void penStart(psfile *out)
  {
    if (t())
      out->gsave();
  }
  
  virtual void penTranslate(psfile *out)
  {
    if (t())
      out->translate(*t() * pair(0,0));
  }

  virtual void penConcat(psfile *out)
  {
    if (t())
      out->concatUnshifted(shiftless(*t()));
  }

  virtual void penEnd(psfile *out)
  {
    if (t())
      out->grestore();
  }
};
  
}

#endif

