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
bbox preclip;
path p;  
public:
  drawClipEnd() {}

  virtual ~drawClipEnd() {}

  void bounds(bbox& b, iopipestream&, std::vector<box>&) {
    b.clip(p.bounds());
    b += preclip;
  }

  bool draw(psfile *out) {
    out->grestore();
    return true;
  }

  drawElement *transformed(const transform& t)
  {
    p=p.transformed(t);
    return drawElement::transformed(t);
  }
  
  friend class drawClipBegin;
};

}

#endif
