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

class drawFill : public drawSuperPathPenBase {
public:
  void noncyclic() {
      reportError("non-cyclic path cannot be filled");
  }
  
  drawFill(vm::array *src, pen pentype)
    : drawSuperPathPenBase(src,pentype) {
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
    out->fill(pentype);
    penEnd(out);
  };

  drawElement *transformed(const transform& t);
};
  
class drawShade : public drawFill {
public:  
  drawShade(vm::array *src, pen pentype)
    : drawFill(src,pentype) {}

  virtual void shade(psfile *out)=0;
  void fill(psfile *out) {
    out->clip(pentype.Fillrule());
    shade(out);
    out->grestore();
  }
};
  
class drawLatticeShade : public drawShade {
protected:
  vm::array *pens;
public:  
  drawLatticeShade(vm::array *src, pen pentype, vm::array *pens)
    : drawShade(src,pentype), pens(pens) {}
  
  void palette(psfile *out) {
    out->gsave();
  }
  
  void shade(psfile *out) {
    out->latticeshade(pens,bpath);
  }
  
  drawElement *transformed(const transform& t);
};
  
class drawAxialShade : public drawShade {
protected:
  pair a;
  pen penb;
  pair b;
  ColorSpace colorspace;
public:  
  drawAxialShade(vm::array *src, pen pentype, pair a, pen penb, pair b) 
    : drawShade(src,pentype), a(a), penb(penb), b(b) {}
  
  void palette(psfile *out);
  
  void shade(psfile *out) {
    out->gradientshade(true,colorspace,pentype,a,0,penb,b,0);
  }
  
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
  
  void shade(psfile *out) {
    out->gradientshade(false,colorspace,pentype,a,ra,penb,b,rb);
  }
  
  drawElement *transformed(const transform& t);
};
  
class drawGouraudShade : public drawShade {
protected:
  vm::array *pens,*vertices,*edges;
public:  
  drawGouraudShade(vm::array *src, pen pentype, vm::array *pens,
		   vm::array *vertices, vm::array *edges)
    : drawShade(src,pentype), pens(pens), vertices(vertices), edges(edges) {}
  
  void palette(psfile *out) {
    out->gsave();
  }
  
  void shade(psfile *out) {
    out->gouraudshade(pens,vertices,edges);
  }
  
  drawElement *transformed(const transform& t);
};
  
class drawTensorShade : public drawShade {
protected:
  vm::array *pens,*boundaries,*z;
public:  
  drawTensorShade(vm::array *src, pen pentype, vm::array *pens,
		  vm::array *boundaries, vm::array *z)
    : drawShade(src,pentype), pens(pens), boundaries(boundaries), z(z) {}
  
  void palette(psfile *out) {
    out->gsave();
  }
  
  void shade(psfile *out) {
    out->tensorshade(pens,boundaries,z);
  }
  
  drawElement *transformed(const transform& t);
};
  
}

#endif
