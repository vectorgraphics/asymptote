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

namespace camp {
  
class drawLabel : public drawElement {
private:
  std::string label;
  double angle;
  pair position;
  pair align;
  pen *pentype;
  pair Align;
  pair adjustment;
  double width,height,depth;
  bool suppress;
  
public:
  drawLabel(std::string label, double angle, pair position, pair align,
	    pen *pentype)
    : label(label), angle(angle), position(position),
      align(align), pentype(pentype), width(0.0), height(0.0), depth(0.0),
      suppress(false) {}
  
  virtual ~drawLabel() {}

  void bounds(bbox& b, iopipestream&, std::vector<box>&);
  
  bool islabel() {
    return true;
  }

  bool write(texfile *out) {
    if(suppress || pentype->transparent()) return true;
    out->setpen(*pentype);
    out->put(label,angle,position+adjustment);
    return true;
  }

  drawElement *transformed(const transform& t);
  
  void labelwarning(const char *action); 
};

}

#endif
