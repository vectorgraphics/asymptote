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

namespace camp {

enum imagetype {PALETTE, NOPALETTE, RAW};
  
class drawImage : public drawElement {
  vm::array image,palette;
  unsigned char *raw; // For internal use; not buffered, may be overwritten.
  size_t width,height;
  transform t;
  bool antialias;
  imagetype type;
public:
  drawImage(const vm::array& image, const vm::array& palette,
            const transform& t, bool antialias, imagetype type=PALETTE)
    : image(image), palette(palette), t(t), antialias(antialias), type(type) {}
  
  drawImage(const vm::array& image, const transform& t, bool antialias)
    : image(image), t(t), antialias(antialias), type(NOPALETTE) {}
  drawImage(unsigned char *raw, size_t width, size_t height, const transform& t,
            bool antialias)
    : raw(raw), width(width), height(height), t(t), antialias(antialias),
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
        out->image(image,palette,antialias);
        break;
      case NOPALETTE:
        out->image(image,antialias);
        break;
      case RAW:
        out->rawimage(raw,width,height,antialias);
        break;
    }
    
    out->grestore();
    
    return true;
  }

  bool svg() {return false;}
  

  drawElement *transformed(const transform& T) {
    return new drawImage(image,palette,T*t,antialias,type);
  }
};

}

#endif
