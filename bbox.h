/*****
 * bbox.h
 * Andy Hammerlindl 2002/06/06
 *
 * Stores a rectangle that encloses a drawing object.
 *****/

#ifndef BBOX_H
#define BBOX_H

#include "pair.h"

extern "C" double round(double x);
  
namespace camp {

inline double min(double a, double b)
{
  return (a < b) ? a : b;
}

inline double max(double a, double b)
{
  return (a > b) ? a : b;
}

inline int min(int a, int b)
{
  return (a < b) ? a : b;
}

inline int max(int a, int b)
{
  return (a > b) ? a : b;
}

// The box that encloses a path
struct bbox {
  bool empty;
  double left;
  double bottom;
  double right;
  double top;
  
  // Start bbox about the origin
  bbox()
    : empty(true), left(0.0), bottom(0.0), right(0.0), top(0.0)
  {
  }

  bbox(double left, double bottom, double right, double top)
    : empty(false), left(left), bottom(bottom), right(right), top(top)
  {
  }

  bbox(pair p, pair q) 
  {
    bbox();
    add(p); add(q);
  }
  
  // Start a bbox with a point
  bbox(pair z)
  {
    bbox();
    add(z);
  }

  bool nonempty() const {
    return !empty;
  }
 
  // Add a point to a bbox
  bbox add(const pair& z)
  {
    double x = z.getx(), y = z.gety();

    if (empty) {
      left = right = x;
      top = bottom = y;
      empty = false;
    }
    else {
      if (x < left)
	left = x;  
      if (x > right)
	right = x;  
      if (y < bottom)
	bottom = y;
      if (y > top)
	top = y;
    }

    return *this;
  }

  bbox operator+= (const pair& z)
  {
    return add(z);
  }

  bbox operator*= (double x)
  {
    left *= x;
    right *= x;
    top *= x;
    bottom *=x;
    return *this;
  }

  // Add two bounding boxes
  friend bbox operator+ (const bbox& b1, const bbox& b2)
  {
    if (b1.empty)
      return b2;
    else if (b2.empty)
      return b1;
    else
      return bbox(min(b1.left, b2.left),
	          max(b1.right, b2.right),
		  min(b1.bottom, b2.bottom),
		  max(b1.top, b2.top));
  }

  // Add one bounding box to another
  bbox add(const bbox& b)
  {
    if (this->empty)
      *this = b;
    else if (!b.empty) {
      left = min(left, b.left);
      right = max(right, b.right);
      bottom = min(bottom, b.bottom);
      top = max(top, b.top);
    }

    return *this;
  }

  bbox operator+= (const bbox& b)
  {
    return add(b);
  }

  void clip(const bbox& b) {
    left = max(left, b.left);
    right = min(right, b.right);
    bottom = max(bottom, b.bottom);
    top = min(top, b.top);
  }
  
  void shift(const pair& p) {
    left += p.getx();
    right += p.getx();
    bottom += p.gety();
    top += p.gety();
  }
  
  pair Min() const {
    return pair(left,bottom);
  }
  
  pair Max() const {
    return pair(right,top);
  }
  
  bbox LowRes()
  {
    return bbox(floor(left),floor(bottom),ceil(right),ceil(top));
  }
  
  friend ostream& operator<< (ostream& out, const bbox& b)
  {
    out << b.left << " " << b.bottom << " " << b.right << " " << b.top;
    return out;
  }
};

// Add results of both bounding boxes, say for a path and a pen.
inline bbox pad(bbox b1, bbox b2)
{
  if (b1.empty)
    return b2;
  else if (b2.empty)
    return b1;
  else {
    bbox b;
    b.empty = false;
    b.left = b1.left + b2.left;
    b.right = b1.right + b2.right;
    b.top = b1.top + b2.top;
    b.bottom = b1.bottom + b2.bottom;
    return b;
  }
}

} // namespace camp

#endif
