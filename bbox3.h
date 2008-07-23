/*****
 * bbox.h
 * Andy Hammerlindl 2002/06/06
 *
 * Stores a rectangle that encloses a drawing object.
 *****/

#ifndef BBOX3_H
#define BBOX3_H

#include "triple.h"

namespace camp {

// The box that encloses a path
struct bbox3 {
  bool empty;
  double left;
  double bottom;
  double right;
  double top;
  double lower;
  double upper;
  
  // Start bbox about the origin
  bbox3()
    : empty(true), left(0.0), bottom(0.0), right(0.0), top(0.0),
      lower(0.0), upper(0.0)
  {
  }

  // Start a bbox with a point
  bbox3(const triple& z)
    : empty(false), left(z.getx()), bottom(z.gety()),
      right(z.getx()), top(z.gety())
  {
  }

  // Add a point to a bbox
  bbox3 add(const triple& v)
  {
    double x = v.getx(), y = v.gety(), z = v.getz();

    if (empty) {
      left = right = lower = x;
      top = bottom = upper =y;
      empty = false;
    }
    else {
      if(x < left)
        left = x;  
      else if(x > right)
        right = x;  
      if(y < bottom)
        bottom = y;
      else if(y > top)
        top = y;
      if(z < lower)
        lower = z;
      else if(z > upper)
        upper = z;
    }

    return *this;
  }

  bbox3 operator+= (const triple& z)
  {
    return add(z);
  }

  triple Min() const {
    return triple(left,bottom,lower);
  }
  
  triple Max() const {
    return triple(right,top,upper);
  }
  
};

} // namespace camp

GC_DECLARE_PTRFREE(camp::bbox3);

#endif
