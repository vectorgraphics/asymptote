/*****
 * drawimage.h
 * John Bowman
 *
 * Stores a image that has been added to a picture.
 *****/

#ifndef DRAWIMAGE_H
#define DRAWIMAGE_H

#include "drawelement.h"
#include "inst.h"
#include "pen.h"

namespace camp {

class drawImage : public drawElement {
  vm::array image,palette;
  transform t;
public:
  drawImage(vm::array image, vm::array palette, const transform& t)
    : image(image), palette(palette), t(t) {}
  
  virtual ~drawImage() {}

  void bounds(bbox& b, iopipestream&, boxvector&, bboxlist&) {
    b += t*pair(0,0);
    b += t*pair(1,1);
  }

  bool draw(psfile *out) {
    out->gsave();
    out->concat(t);
    out->image(&image,&palette);
    out->grestore();
    
    return true;
  }

  drawElement *transformed(const transform& T) {
    return new drawImage(image,palette,T*t);
  }
};

}

#endif
