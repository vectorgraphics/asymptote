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

enum Shading {NONE,AXIAL,RADIAL,GOURAUD};
  
class drawFill : public drawSuperPathPenBase {
public:
  void noncyclic() {
      reportError("non-cyclic path cannot be filled");
  }
  
  drawFill(vm::array *src, pen pentype) : drawSuperPathPenBase(src,pentype) {
    if(!cyclic()) noncyclic();
  }

  virtual ~drawFill() {}

  bool draw(psfile *out);
  virtual void palette(psfile *out) {
    out->setpen(pentype);
    penStart(out);
    penTranslate(out);
  }
  virtual void fill(psfile *out) {
    out->fill(pentype.Fillrule());
    penEnd(out);
  };

  drawElement *transformed(const transform& t);
};
  
class drawAxialShade : public drawFill {
protected:
  pair a;
  pen penb;
  pair b;
  ColorSpace colorspace;
public:  
  drawAxialShade(vm::array *src, pen pentype, pair a, pen penb, pair b)
    : drawFill(src,pentype), a(a), penb(penb), b(b) {}
  
  void palette(psfile *out);
  void fill(psfile *out);
  
  drawElement *transformed(const transform& t);
};
  
class drawRadialShade : public drawAxialShade {
protected:
  double ra;
  double rb;
public:
  drawRadialShade(vm::array *src,
	   pen pentype, pair a, double ra, pen penb, pair b, double rb)
    : drawAxialShade(src,pentype,a,penb,b), ra(ra), rb(rb) {}
  
  void fill(psfile *out);
  
  drawElement *transformed(const transform& t);
};
  
}

#endif
