/*****
 * drawfill.h
 * Andy Hammerlindl 2002/06/06
 *
 * Stores a cyclic path that will outline a filled shape in a picture.
 *****/

#ifndef DRAWFILL_H
#define DRAWFILL_H

#include "drawelement.h"
#include "path.h"

namespace camp {

class drawFill : public drawPathPenBase {
  pair a;
  double ra;
  pen penb;
  pair b;
  double rb;
  bool axial;
  bool shade;
  ColorSpace colorspace;
public:
  void noncyclic() {
      reportError("non-cyclic path cannot be filled");
  }
  
  drawFill(path src,
	   pen pentype, pair a, double ra, pen penb, pair b, double rb)
    : drawPathPenBase(src, pentype), a(a), ra(ra), penb(penb), b(b), rb(rb) {
    if(!cyclic()) noncyclic();
  }

  drawFill(vm::array *src,
	   pen pentype, pair a, double ra, pen penb, pair b, double rb)
    : drawPathPenBase(src, pentype), a(a), ra(ra), penb(penb), b(b), rb(rb) {
    if(!cyclic()) noncyclic();
  }

  virtual ~drawFill() {}

  bool draw(psfile *out);
  void palette(psfile *out);
  void fill(psfile *out);

  drawElement *transformed(const transform& t);
};
  
}

#endif
