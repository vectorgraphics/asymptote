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
    if(bboxstack.size() < 2)
      reportError("endclip without matching beginclip");
    b.clip(bboxstack.back());
    bboxstack.pop_back();
    b += bboxstack.back();
    bboxstack.pop_back();
  }

  bool endgroup() {return true;}

  bool svg() {return true;}

  void save(bool b) {
    grestore=b;
    if(partner) partner->save(b);
  }

  bool draw(psfile *out) {
    if(grestore) out->grestore();
    return true;
  }

  bool write(texfile *out, const bbox& bpath) {
    out->endgroup();

    if(out->toplevel())
      out->endpicture(bpath);

    if(grestore) out->grestore();
    return true;
  }
};

class drawClip3End : public drawElement {
  drawClip3Begin *partner;
public:
  drawClip3End(drawClip3Begin *partner=NULL) :
    partner(partner) {}

  virtual ~drawClip3End() {}

  bool endclip() {return true;}

  bool endgroup() {return true;}

  void render(double size2, const triple& Min, const triple& Max,
              double perspective, bool remesh) {
    clipStack.pop_back();
    --clipStackStackIndex;
  }
};

}

GC_DECLARE_PTRFREE(camp::drawClipEnd);

#endif
