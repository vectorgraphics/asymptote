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
  const unsigned char *raw; // For internal use; not buffered.
  size_t width,height;
  ColorSpace colorspace;
  transform t;
  imagetype type;
public:
  drawImage(const vm::array& image, const vm::array& palette,
	    const transform& t)
    : image(image), palette(palette), t(t), type(PALETTE) {}
  
  drawImage(const vm::array& image, const transform& t)
    : image(image), t(t), type(NOPALETTE) {}
  drawImage(const unsigned char *raw, size_t width, size_t height,
	    ColorSpace colorspace, const transform& t)
    : raw(raw), width(width), height(height), colorspace(colorspace), t(t),
      type(RAW) {}
  
  
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
      out->image(image,palette);
      break;
    case NOPALETTE:
      out->image(image);    
      break;
    case RAW:
      out->image(raw,width,height,colorspace);
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
