/*****
 * drawpath.h
 * Andy Hammerlindl 2002/06/06
 *
 * Stores a path that has been added to a picture.
 *****/

#ifndef DRAWPATH_H
#define DRAWPATH_H

#include "drawelement.h"
#include "path.h"

namespace camp {

class drawPath : public drawPathPenBase {
public:
  drawPath(path src, pen pentype) : drawPathPenBase(src, pentype) {}
  
  virtual ~drawPath() {}

  // Account for pen cap contribution to bounding box.
  void addcap(bbox& b, const path& p, double t, const pair& dir);
    
  void bounds(bbox& b, iopipestream&, boxvector&, bboxlist&);

  bool draw(psfile *out);

  virtual void adjustdash(pen &);

  drawElement *transformed(const transform& t);
};

}

#endif
