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
public:
  drawFill(path src, pen pentype, pair a, double ra, pen penb, pair b,
	   double rb) : drawPathPenBase(src, pentype), a(a), ra(ra),
			penb(penb), b(b), rb(rb) {
    if (!p.cyclic())
      reportError("non-cyclic path cannot be filled");
  }

  virtual ~drawFill() {}

  bool draw(psfile *out);

  drawElement *transformed(const transform& t);
};

}

#endif
