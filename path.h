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
#include <cfloat>

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

inline void checkEmpty(int n) {
  if(n == 0)
    reportError("nullpath has no points");
}

inline int adjustedIndex(int i, int n, bool cycles)
{
  checkEmpty(n);
  if(cycles)
    return imod(i,n);
  else if(i < 0)
    return 0;
  else if(i >= n)
    return n-1;
  else
    return i;
}

// Used in the storage of solved path knots.
struct solvedKnot : public gc {
  pair pre;
  pair point;
  pair post;
  bool straight;
  solvedKnot() : straight(false) {}
  
  friend bool operator== (const solvedKnot& p, const solvedKnot& q)
  {
    return p.pre == q.pre && p.point == q.point && p.post == q.post;
  }
};

extern const double Fuzz;
extern const double Fuzz2;
extern const double sqrtFuzz;
  
class path : public gc {
  bool cycles;  // If the path is closed in a loop

  int n; // The number of knots

  mem::vector<solvedKnot> nodes;
  mutable double cached_length; // Cache length since path is immutable.
  mutable bbox box;

public:
  path()
    : cycles(false), n(0), nodes(), cached_length(-1) {}

  // Create a path of a single point
  path(pair z,bool = false)
    : cycles(false), n(1), nodes(1), cached_length(-1)
  {
    nodes[0].pre = nodes[0].point = nodes[0].post = z;
    nodes[0].straight = false;
  }  

  // Creates path from a list of knots.  This will be used by camp
  // methods such as the guide solver, but should probably not be used by a
  // user of the system unless he knows what he is doing.
  path(mem::vector<solvedKnot>& nodes, int n, bool cycles = false)
    : cycles(cycles), n(n), nodes(nodes), cached_length(-1)
  {
  }

  friend bool operator== (const path& p, const path& q)
  {
    return p.cycles == q.cycles && p.nodes == q.nodes;
  }

private:
  path(solvedKnot n1, solvedKnot n2)
    : cycles(false), n(2), nodes(2), cached_length(-1)
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
  
  mem::vector<solvedKnot>& Nodes() {
    return nodes;
  }
  
  bool straight(int t) const
  {
    if (cycles) return nodes[imod(t,n)].straight;
    return (t >= 0 && t < n) ? nodes[t].straight : false;
  }
  
  pair point(int t) const
  {
    return nodes[adjustedIndex(t,n,cycles)].point;
  }

  pair point(double t) const;
  
  pair precontrol(int t) const
  {
    return nodes[adjustedIndex(t,n,cycles)].pre;
  }

  pair precontrol(double t) const;
  
  pair postcontrol(int t) const
  {
    return nodes[adjustedIndex(t,n,cycles)].post;
  }

  pair postcontrol(double t) const;
  
  pair predir(int t) const {
    if(!cycles && t <= 0) return pair(0,0);
    pair z0=point(t-1);
    pair z1=point(t);
    pair c1=precontrol(t);
    pair dir=z1-c1;
    double epsilon=Fuzz2*(z0-z1).abs2();
    if(dir.abs2() > epsilon) return unit(dir);
    pair c0=postcontrol(t-1);
    dir=2*c1-c0-z1;
    if(dir.abs2() > epsilon) return unit(dir);
    return unit(z1-z0+3*(c0-c1));
  }

  pair predir(double t) const {
    if(!cycles) {
      if(t <= 0) return pair(0,0);
      if(t >= n-1) return predir(n-1);
    }
    int a=Floor(t);
    return (t-a < sqrtFuzz) ? predir(a) : subpath((double) a,t).predir(1);
  }

  pair postdir(int t) const {
    if(!cycles && t >= n-1) return pair(0,0);
    pair z0=point(t);
    pair z1=point(t+1);
    pair c0=postcontrol(t);
    pair dir=c0-z0;
    double epsilon=Fuzz2*(z0-z1).abs2();
    if(dir.abs2() > epsilon) return unit(dir);
    pair c1=precontrol(t+1);
    dir=z0-2*c0+c1;
    if(dir.abs2() > epsilon) return unit(dir);
    return unit(z1-z0+3*(c0-c1));
  }

  pair postdir(double t) const {
    if(!cycles) {
      if(t >= n-1) return pair(0,0);
      if(t <= 0) return postdir(0);
    }
    int b=Ceil(t);
    return (b-t < sqrtFuzz) ? postdir(b) : subpath(t,(double) b).postdir(0);
  }

  pair dir(int t) const {
    return unit(predir(t)+postdir(t));
  }
  
  pair dir(double t) const {
    return unit(predir(t)+postdir(t));
  }

  pair dir(int t, int sign) const {
    if(sign == 0) return dir(t);
    else if(sign > 0) return postdir(t);
    else return predir(t);
  }

  // Returns the path traced out in reverse.
  path reverse() const;

  // Generates a path that is a section of the old path, using the time
  // interval given.
  path subpath(int start, int end) const;
  path subpath(double start, double end) const;

  // Used by picture to determine bounding box
  bbox bounds() const;
  
  // Return bounding box accounting for padding perpendicular to path.
  bbox bounds(double min, double max) const;

  // Return bounding box accounting for internal pen padding (but not pencap).
  bbox internalbounds(const bbox &padding) const;
  
  double arclength () const;
  double arctime (double l) const;
  double directiontime(pair z) const;
 
  pair max() {
    return bounds().Max();
  }

  pair min() {
    return bounds().Min();
  }
  
  // Debugging output
  friend std::ostream& operator<< (std::ostream& out, const path p);

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

extern path nullpath;
  
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

GC_DECLARE_PTRFREE(camp::solvedKnot);
GC_DECLARE_PTRFREE(camp::path);

#endif
