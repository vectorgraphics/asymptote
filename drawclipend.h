/*****
 * drawclipend.h
 * John Bowman
 *
 * End clip of picture to specified path.
 *****/

#ifndef DRAWCLIPEND_H
#define DRAWCLIPEND_H

#include "drawelement.h"
#include "path.h"

namespace camp {

class drawClipEnd : public drawElement {
bool grestore;  
public:
  drawClipEnd(bool grestore=true) : grestore(grestore) {}

  virtual ~drawClipEnd() {}

  void bounds(bbox& b, iopipestream&, std::vector<box>&) {
    if(bboxstack.size() < 2) {
      reportError("endclip without matching beginclip");
      return;
    }
    b.clip(bboxstack.back());
    bboxstack.pop_back();
    b += bboxstack.back();
    bboxstack.pop_back();
  }

  bool draw(psfile *out) {
    if(grestore) out->grestore();
    return true;
  }

  friend class drawClipBegin;
};

}

#endif
