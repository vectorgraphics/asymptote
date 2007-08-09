/*****
 * drawclipend.h
 * John Bowman
 *
 * End clip of picture to specified path.
 *****/

#ifndef DRAWCLIPEND_H
#define DRAWCLIPEND_H

#include "drawclipbegin.h"
#include "path.h"

namespace camp {

class drawClipEnd : public drawElement {
  bool grestore;  
  drawClipBegin *partner;
public:
  drawClipEnd(bool grestore=true, drawClipBegin *partner=NULL) : 
    grestore(grestore), partner(partner) {}

  virtual ~drawClipEnd() {}

  bool endclip() {return true;}
  
  void bounds(bbox& b, iopipestream&, boxvector&, bboxlist& bboxstack) {
    if(bboxstack.size() < 2) {
      reportError("endclip without matching beginclip");
    }
    b.clip(bboxstack.back());
    bboxstack.pop_back();
    b += bboxstack.back();
    bboxstack.pop_back();
  }

  bool endgroup() {return true;}
  
  void save(bool b) {
    grestore=b;
    partner->save(b);
  }
  
  bool draw(psfile *out) {
    if(grestore) out->grestore(false,true);
    return true;
  }

  bool write(texfile *out) {
    if(grestore) out->grestore(true);
    return true;
  }

};

}

GC_DECLARE_PTRFREE(camp::drawClipEnd);

#endif
