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

class drawClipBegin : public drawPathBase {
public:
  // The bounding box before the clipping.
  drawClipEnd &partner;

public:
  drawClipBegin(path src, drawClipEnd& partner) : drawPathBase(src),
						  partner(partner) {
    if (!p.cyclic())
      reportError("cannot clip to non-cyclic path");
  }

  virtual ~drawClipBegin() {}

  void bounds(bbox& b, iopipestream&, std::vector<box>&) {
    partner.preclip=b;
    partner.p=p;
  }

  bool draw(psfile *out) {
    out->gsave();
    if (!p.empty()) {
      out->write(p);
      out->clip();
    }
    return true;
  }

  drawElement *transformed(const transform& t)
  {
    return new drawClipBegin(transpath(t),partner);
  }

};

// Adds the drawElements to clip a picture to a path.
// Subsequent additions to the picture will not be affected by the path.
inline void clip(picture &pic, path p)
{
  drawClipEnd *e = new drawClipEnd();
  drawClipBegin *b = new drawClipBegin(p,*e);
  pic.prepend(b);
  pic.append(e);
}

}

#endif
