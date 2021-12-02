/*****
 * drawlayer.h
 * John Bowman
 *
 * Start a new postscript/TeX layer in picture.
 *****/

#ifndef DRAWLAYER_H
#define DRAWLAYER_H

#include "drawelement.h"

namespace camp {

class drawLayer : public drawElement {
public:
  drawLayer() {}

  virtual ~drawLayer() {}

  bool islayer() {return true;}
};

class drawNewPage : public drawLayer {
  bbox box;
public:
  drawNewPage() : box() {}
  drawNewPage(const bbox& box) : box(box) {}

  virtual ~drawNewPage() {}

  bool islabel() {return true;}
  bool isnewpage() {return true;}

  bool write(texfile *out, const bbox& b) {
    out->newpage(box.empty ? b : box);
    return true;
  }
};

}

GC_DECLARE_PTRFREE(camp::drawLayer);

#endif
