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

namespace camp {

class drawClipBegin : public drawPathPenBase {
bool gsave;
public:
  void noncyclic() {
      reportError("cannot clip to non-cyclic path");
  }
  
  drawClipBegin(path src, const pen& pentype, bool gsave=true)
    : drawPathPenBase(src,pentype), gsave(gsave) {
    if(!cyclic()) noncyclic();
  }

  drawClipBegin(vm::array *src, const pen& pentype, bool gsave=true)
    : drawPathPenBase(src,pentype), gsave(gsave) {
    for(size_t i=0; i < size; i++)
      if(!cyclic()) noncyclic();
  }

  virtual ~drawClipBegin() {}

  void bounds(bbox& b, iopipestream& iopipe, boxvector& vbox,
	      bboxlist& bboxstack) {
    bboxstack.push_back(b);
    bbox bpath;
    drawPathPenBase::bounds(bpath,iopipe,vbox,bboxstack);
    bboxstack.push_back(bpath);
  }

  bool begingroup() {return true;}
  
  bool draw(psfile *out) {
    if(gsave) out->gsave();
    if (empty()) return true;
    
    writepath(out);
    out->clip(pentype.Fillrule());
    return true;
  }

  drawElement *transformed(const transform& t)
  {
    if(P)
      return new drawClipBegin(transPath(t),transpen(t));
    else
      return new drawClipBegin(transpath(t),transpen(t));
  }

};

}

#endif
