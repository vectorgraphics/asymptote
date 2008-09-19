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

enum imagetype {PALETTE, NOPALETTE, RAW};
  
class drawImage : public drawElement {
  vm::array image,palette;
   unsigned char *raw; // For internal use; not buffered.
  size_t width,height;
  transform t;
  imagetype type;
public:
  drawImage(const vm::array& image, const vm::array& palette,
	    const transform& t)
    : image(image), palette(palette), t(t), type(PALETTE) {}
  
  drawImage(const vm::array& image, const transform& t)
    : image(image), t(t), type(NOPALETTE) {}
  drawImage(unsigned char *raw, size_t width, size_t height,
	    const transform& t)
    : raw(raw), width(width), height(height), t(t), type(RAW) {}
  
  
  virtual ~drawImage() {}

  void bounds(bbox& b, iopipestream&, boxvector&, bboxlist&) {
    b += t*pair(0,0);
    b += t*pair(1,1);
  }

  bool draw(psfile *out) {
    out->gsave();
    out->concat(t);
    switch(type) {
    case PALETTE:
      out->image(image,palette,false);
      break;
    case NOPALETTE:
      out->image(image,false);    
      break;
    case RAW:
      out->rawimage(raw,width,height,true);
      break;
    }
    
    out->grestore();
    
    return true;
  }

  drawElement *transformed(const transform& T) {
    return new drawImage(image,palette,T*t);
  }
};

}

#endif
