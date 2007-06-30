/*****
 * drawimage.h
 * John Bowman
 *
 * Stores a image that has been added to a picture.
 *****/

#ifndef DRAWIMAGE_H
#define DRAWIMAGE_H

#include "drawelement.h"
#include "array.h"
#include "pen.h"

namespace camp {

class drawImage : public drawElement {
  vm::array image,palette;
  transform t;
  bool havepalette;
public:
  drawImage(const vm::array& image, const vm::array& palette,
	    const transform& t)
    : image(image), palette(palette), t(t), havepalette(true) {}
  
  drawImage(const vm::array& image, const transform& t)
    : image(image), t(t), havepalette(false) {}
  
  
  virtual ~drawImage() {}

  void bounds(bbox& b, iopipestream&, boxvector&, bboxlist&) {
    b += t*pair(0,0);
    b += t*pair(1,1);
  }

  bool draw(psfile *out) {
    out->gsave();
    out->concat(t);
    if(havepalette) out->image(image,palette);
    else out->image(image);
    out->grestore();
    
    return true;
  }

  drawElement *transformed(const transform& T) {
    return new drawImage(image,palette,T*t);
  }
};

}

GC_DECLARE_PTRFREE(camp::drawImage);

#endif
