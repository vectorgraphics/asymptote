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
#include "pool.h"

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
  
class drawElement : public mempool::pooled<drawElement>
{
public:
  // Adjust the bbox of the picture based on the addition of this
  // element. The iopipestream is needed for determining label sizes.
  virtual void bounds(bbox&, iopipestream&, std::vector<box>&) {}

  virtual bool islabel() {
    return false;
  }

  virtual bool islayer() {
    return false;
  }

  // Do any initial set up
  virtual bool setup(texfile *) {
    return true;
  }

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
  drawPathBase(path p) : p(p) {}

  virtual ~drawPathBase() {}

  void bounds(bbox& b, iopipestream&, std::vector<box>&) {
    b += p.bounds();
  }
};

// Base class for drawElements that involve paths and pens.
class drawPathPenBase : public drawPathBase {
protected:
  pen pentype;

  // The pen's transform;
  const transform *t()
  {
    return pentype.getTransform();
  }

  pen transpen(const transform& t) {
    return camp::transformed(new transform(shiftless(t)),pentype);
  }

public:
  drawPathPenBase(path p, pen pentype) : drawPathBase(p), pentype(pentype) {}

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
      out->concat(shiftless(*t()));
  }

  virtual void penEnd(psfile *out)
  {
    if (t())
      out->grestore();
  }
};
  
}

#endif

