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
public:
  drawBegin() {}
  
  virtual ~drawBegin() {}

  bool begingroup() {return true;}
};
  
class drawEnd : public drawElement {
public:
  drawEnd() {}
  
  virtual ~drawEnd() {}

  bool endgroup() {return true;}
};

}

#endif
