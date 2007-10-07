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

class drawClipBegin : public drawSuperPathPenBase {
  bool gsave;
public:
  void noncyclic() {
      reportError("cannot clip to non-cyclic path");
  }
  
  drawClipBegin(const vm::array& src, pen pentype, bool gsave=true)
    : drawSuperPathPenBase(src,pentype), gsave(gsave) {
    for(size_t i=0; i < size; i++)
      if(!cyclic()) noncyclic();
  }

  virtual ~drawClipBegin() {}

  void bounds(bbox& b, iopipestream& iopipe, boxvector& vbox,
	      bboxlist& bboxstack) {
    bboxstack.push_back(b);
    bbox bpath;
    drawSuperPathPenBase::bounds(bpath,iopipe,vbox,bboxstack);
    bboxstack.push_back(bpath);
  }

  bool begingroup() {return true;}
  
  void save(bool b) {
    gsave=b;
  }
  
  bool draw(psfile *out) {
    if(gsave) out->gsave();
    if(empty()) return true;
    
    writepath(out);
    out->clip(pentype);
    return true;
  }

  bool write(texfile *out, const bbox& bpath) {
    if(gsave) out->gsave();
    if(empty()) return true;
    
    out->verbatim("\\begin{picture}(");
    double width=bpath.right-bpath.left;
    double height=bpath.top-bpath.bottom;
    out->write(width*ps2tex);
    out->verbatim(",");
    out->write(height*ps2tex);
    out->verbatimline(")%");

    out->beginspecial();
    out->beginraw();
    writeshiftedpath(out);
    out->clip(pentype);
    out->endraw();
    out->endspecial();
    
    return true;
  }
  
  drawElement *transformed(const transform& t)
  {
    return new drawClipBegin(transpath(t),transpen(t));
  }

};

}

#endif
