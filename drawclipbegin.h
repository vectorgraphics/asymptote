/*****
 * drawclipbegin.h
 * John Bowman
 *
 * Begin clip of picture to specified path.
 *****/

#ifndef DRAWCLIPBEGIN_H
#define DRAWCLIPBEGIN_H

#include "drawelement.h"
#include "path.h"
#include "drawpath.h"
#include "drawclipend.h"

namespace camp {

class drawClipBegin : public drawPathPenBase {
public:
  // The bounding box before the clipping.
  drawClipEnd &partner;

public:
  void noncyclic() {
      reportError("cannot clip to non-cyclic path");
  }
  
  drawClipBegin(path src, const pen& pentype, drawClipEnd& partner) 
    : drawPathPenBase(src,pentype), partner(partner) {
    if(!cyclic()) noncyclic();
  }

  drawClipBegin(vm::array *src, const pen& pentype, drawClipEnd& partner) 
    : drawPathPenBase(src,pentype), partner(partner) {
    for(size_t i=0; i < size; i++)
      if(!cyclic()) noncyclic();
  }

  virtual ~drawClipBegin() {}

  void bounds(bbox& b, iopipestream& iopipe, std::vector<box>& vbox) {
    partner.preclip=b;
    drawPathPenBase::bounds(partner.Bounds,iopipe,vbox);
  }

  bool draw(psfile *out) {
    out->gsave();
    
    if (empty()) return true;
    writepath(out);
    out->clip(pentype.Fillrule());

    return true;
  }

  drawElement *transformed(const transform& t)
  {
    if(P)
      return new drawClipBegin(transPath(t),transpen(t),partner);
    else
      return new drawClipBegin(transpath(t),transpen(t),partner);
  }

};

// Adds the drawElements to clip a picture to a path.
// Subsequent additions to the picture will not be affected by the path.
inline void clip(picture &pic, path p, const pen &pentype)
{
  drawClipEnd *e = new drawClipEnd();
  drawClipBegin *b = new drawClipBegin(p,pentype,*e);
  pic.prepend(b);
  pic.append(e);
}

inline void clip(picture &pic, vm::array *P, const pen &pentype)
{
  drawClipEnd *e = new drawClipEnd();
  drawClipBegin *b = new drawClipBegin(P,pentype,*e);
  pic.prepend(b);
  pic.append(e);
}

}

#endif
