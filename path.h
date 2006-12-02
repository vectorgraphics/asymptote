/*****
 * path.h
 * Andy Hammerlindl 2002/05/16
 *
 * Stores a piecewise cubic spline with known control points.
 *
 * When changing the path algorithms, also update the corresponding 
 * three-dimensional algorithms in path3.cc and three.asy.
 *****/

#ifndef PATH_H
#define PATH_H

#include <iostream>
#include <climits>

#include "mod.h"
#include "pair.h"
#include "transform.h"
#include "bbox.h"

inline double intcap(double t) {
  if(t <= -INT_MAX) return -INT_MAX;
  if(t >= INT_MAX) return INT_MAX;
    return t;
}
  
// The are like floor and ceil, except they return an integer;
// if the argument cannot be converted to a valid integer, they return
// INT_MAX (for positive arguments) or -INT_MAX (for negative arguments).

inline int Floor(double t) {return (int) floor(intcap(t));}
inline int Ceil(double t) {return (int) ceil(intcap(t));}

bool simpson(double& integral, double (*)(double), double a, double b,
	     double acc, double dxmax);

bool unsimpson(double integral, double (*)(double), double a, double& b,
	       double acc, double& area, double dxmax);

namespace camp {

using std::ostream;
using std::endl;

// Used in the storage of solved path knots.
struct solvedKnot : public gc {
  pair pre;
  pair point;
  pair post;
  bool straight;
  solvedKnot() : straight(false) {}
};


class path : public gc {
  bool cycles;  // If the knot is closed in a loop

  int n; // The number of knots

  solvedKnot *nodes;
  mutable double cached_length; // Cache length since path is immutable.
  mutable bbox box;

public:
  path()
    : cycles(false), n(0), nodes(), cached_length(-1) {}

  // Create a path of a single point
  path(pair z,bool = false)
    : cycles(false), n(1), nodes(new solvedKnot[n]), cached_length(-1)
  {
    nodes[0].pre = nodes[0].point = nodes[0].post = z;
    nodes[0].straight = false;
  }  

  // Creates path from a list of knots.  This will be used by camp
  // methods such as the guide solver, but should probably not be used by a
  // user of the system unless he knows what he is doing.
  path(solvedKnot *nodes, int n, bool cycles = false)
    : cycles(cycles), n(n), nodes(nodes), cached_length(-1)
  {
  }

private:
  path(solvedKnot n1, solvedKnot n2)
    : cycles(false), n(2), nodes(new solvedKnot[n]) ,cached_length(-1)
  {
    nodes[0] = n1;
    nodes[1] = n2;
    nodes[0].pre = nodes[0].point;
    nodes[1].post = nodes[1].point;
  }
public:

  // Copy constructor
  path(const path& p)
    : cycles(p.cycles), n(p.n), nodes(p.nodes), cached_length(p.cached_length),
      box(p.box)
  {}

  virtual ~path()
  {
  }

  // Getting control points
  int size() const
  {
    return n;
  }

  bool empty() const
  {
    return n == 0;
  }

  int length() const
  {
    return cycles ? n : n-1;
  }

  bool cyclic() const
  {
    return cycles;
  }
  
  solvedKnot *Nodes() {
    return nodes;
  }
  
  void emptyError() const {
    if(empty())
      reportError("nullpath has no points");
  }
  
  pair point(int i) const
  {
    emptyError();
    
    if (cycles)
      return nodes[imod(i,n)].point;
    else if (i < 0)
      return nodes[0].point;
    else if (i >= n)
      return nodes[n-1].point;
    else
      return nodes[i].point;
  }

  bool straight(int i) const
  {
    if (cycles) return nodes[imod(i,n)].straight;
    return (i >= 0 && i < n) ? nodes[i].straight : false;
  }
  
  pair point(double t) const;
  
  pair precontrol(int i) const
  {
    emptyError();
		       
    if (cycles)
      return nodes[imod(i,n)].pre;
    else if (i < 0)
      return nodes[0].pre;
    else if (i >= n)
      return nodes[n-1].pre;
    else
      return nodes[i].pre;
  }

  pair precontrol(double t) const;
  
  pair postcontrol(int i) const
  {
    emptyError();
		       
    if (cycles)
      return nodes[imod(i,n)].post;
    else if (i < 0)
      return nodes[0].post;
    else if (i >= n)
      return nodes[n-1].post;
    else
      return nodes[i].post;
  }

  pair postcontrol(double t) const;
  
  pair direction(int i) const
  {
    return postcontrol(i) - precontrol(i);
  }

  pair direction(double t) const
  {
    return postcontrol(t) - precontrol(t);
  }

  // Returns the path traced out in reverse.
  path reverse() const;

  // Generates a path that is a section of the old path, using the time
  // interval given.
  path subpath(int start, int end) const;
  path subpath(double start, double end) const;

  // Used by picture to determine bounding box
  // NOTE: Conservative here; uses only control points.
  //       A better method is in the MetaPost source.
  bbox bounds() const;
  
  // Return bounding box accounting for internal pen padding (but not pencap).
  bbox bounds(const bbox &padding) const;

  double arclength () const;
  double arctime (double l) const;
  double directiontime(pair z) const;
 
  pair max () {
    return bounds().Max();
  }

  pair min () {
    return bounds().Min();
  }
  
  // Debugging output
  friend ostream& operator<< (ostream& out, const path p);

  int sgn1(double x) const
  {
    return x > 0.0 ? 1 : -1;
  }

// Increment count if the path has a vertical component at t.
  bool Count(int& count, double t) const;
  
// Count if t is in (begin,end] and z lies to the left of point(i+t).
  void countleft(int& count, double x, int i, double t,
		 double begin, double end, double& mint, double& maxt) const;

// Return the winding number of the region bounded by the (cyclic) path
// relative to the point z.
  int windingnumber(const pair& z) const;

  // Transformation
  path transformed(const transform& t) const;
  
};

bool intersect(pair& t, path p1, path p2, double fuzz);
  
// Concatenates two paths into a new one.
path concat(path p1, path p2);

// Applies a transformation to the path
path transformed(const transform& t, path p);
  
inline double quadratic(double a, double b, double c, double x)
{
  return a*x*x+b*x+c;
}
  
class quadraticroots {
public:
  enum {NONE=0, ONE=1, TWO=2, MANY} distinct; // Number of distinct roots.
  unsigned int roots; // Total number of real roots.
  double t1,t2;
  
  quadraticroots(double a, double b, double c);
};

class cubicroots {
public:  
  unsigned int roots; // Total number of real roots.
  double t1,t2,t3;
  cubicroots(double a, double b, double c, double d);
};

  
}

#endif
