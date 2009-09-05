/*****
 * drawgroup.h
 * John Bowman
 *
 * Group elements in a picture to be deconstructed as a single object.
 *****/

#ifndef DRAWGROUP_H
#define DRAWGROUP_H

#include "drawelement.h"

namespace camp {

class drawBegin : public drawElement {
  string name;
public:
  drawBegin(string name="") : name(name) {}
  
  virtual ~drawBegin() {}

  bool begingroup() {return true;}
  
  bool write(prcfile *out) {
    out->begingroup(name.c_str());
    return true;
  }

};
  
class drawEnd : public drawElement {
public:
  drawEnd() {}
  
  virtual ~drawEnd() {}

  bool endgroup() {return true;}
  
  bool write(prcfile *out) {
    out->endgroup();
    return true;
  }

};

}

GC_DECLARE_PTRFREE(camp::drawBegin);
GC_DECLARE_PTRFREE(camp::drawEnd);

#endif
