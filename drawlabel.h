/*****
 * drawlabel.h
 * John Bowman 2003/03/14
 *
 * Add a label to a picture.
 *****/

#ifndef DRAWLABEL_H
#define DRAWLABEL_H

#include "drawelement.h"
#include "path.h"
#include "angle.h"
#include "transform.h"

namespace camp {
  
class drawLabel : public drawElement {
private:
  string label,size;
  transform T;		// A linear (shiftless) transformation.
  pair position;
  pair align;
  pair scale;
  pen pentype;
  double width,height,depth;
  bool havebounds;
  bool suppress;
  double fontscale;
  pair Align;
  pair texAlign;
  bbox Box;
  
public:
  drawLabel(string label, string size, transform T, pair position, pair align,
	   pen pentype)
    : label(label), size(size), T(shiftless(T)), position(position),
      align(align), pentype(pentype), width(0.0), height(0.0), depth(0.0),
      havebounds(false), suppress(false), fontscale(1.0) {} 
  
  virtual ~drawLabel() {}

  void bounds(bbox& b, iopipestream&, boxvector&, bboxlist&);
  
  bool texbounds(iopipestream& tex, string& s, bool warn);
    
  bool islabel() {
    return true;
  }

  bool write(texfile *out) {
    if(!havebounds) 
      reportError("drawLabel::write called before bounds");
    if(suppress || pentype.invisible()) return true;
    out->setpen(pentype);
    out->put(label,T,position,texAlign,Box);
    return true;
  }

  drawElement *transformed(const transform& t);
  
  void labelwarning(const char *action); 
};

}

#endif
