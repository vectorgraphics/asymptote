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
  double width,height,depth;
  bool havebounds;
  bool suppress;
  double scale;
  pair Align;
  
public:
  drawLabel(std::string label, double angle, pair position, pair align,
	    pen *pentype)
    : label(label), angle(angle), position(position), align(align),
      pentype(pentype), width(0.0), height(0.0), depth(0.0),
      havebounds(false), suppress(false), scale(1.0) {} 
  
  virtual ~drawLabel() {}

  void bounds(bbox& b, iopipestream&, std::vector<box>&, std::list<bbox>&);
  
  bool islabel() {
    return true;
  }

  bool write(texfile *out) {
    if(!havebounds) 
      reportError("drawLabel::write called before bounds");
    if(suppress || pentype->invisible()) return true;
    out->setpen(*pentype);
    out->put(label,angle,position+Align);
    return true;
  }

  drawElement *transformed(const transform& t);
  
  void labelwarning(const char *action); 
};

}

#endif
