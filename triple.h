/*****
 * triple.h
 * John Bowman
 *
 * Stores a three-dimensional point.
 *
 *****/

#ifndef TRIPLE_H
#define TRIPLE_H

#include <cassert>
#include <iostream>
#include <cmath>

#include "common.h"
#include "angle.h"

namespace camp {

class triple : public gc {
  double x;
  double y;
  double z;

public:
  triple() : x(0.0), y(0.0), z(0.0) {}
  triple(double x, double y=0.0, double z=0.0) : x(x), y(y), z(z) {}

  double getx() const { return x; }
  double gety() const { return y; }
  double getz() const { return z; }

  friend triple operator+ (const triple& z, const triple& w)
  {
    return triple(z.x + w.x, z.y + w.y, z.z + w.z);
  }

  friend triple operator- (const triple& z, const triple& w)
  {
    return triple(z.x - w.x, z.y - w.y, z.z - w.z);
  }

  friend triple operator- (const triple& z)
  {
    return triple(-z.x, -z.y, -z.z);
  }

  friend triple operator* (double s, const triple &z)
  {
    return triple(s*z.x, s*z.y, s*z.z);
  }

  friend triple operator/ (const triple &z, double s)
  {
    if (s == 0.0)
      reportError("division by 0");
    s=1.0/s;
    return triple(z.x*s, z.y*s, z.z*s);
  }

  const triple& operator+= (const triple& w)
  {
    x += w.x;
    y += w.y;
    z += w.z;
    return *this;
  }

  friend bool operator== (const triple& z, const triple& w)
  {
    return z.x == w.x && z.y == w.y && z.z == w.z;
  }

  friend bool operator!= (const triple& z, const triple& w)
  {
    return z.x != w.x || z.y != w.y || z.z != w.z;
  }

  double abs2() const
  {
    return x*x+y*y+z*z;
  }
  
  double length() const /* r */
  {
    return sqrt(abs2());
  }
  
  double polar() const /* theta */
  {
    double r=length();
    if (r == 0.0)
      reportError("taking polar angle of (0,0,0)");
    return acos(z/r);
  }
  
  double azimuth() const /* phi */
  {
    return angle(x,y);
  }
  
  friend triple unit(const triple& z)
  {
    double scale=z.length();
    if(scale != 0.0) scale=1.0/scale;
    return triple(z.x*scale,z.y*scale,z.z*scale);
  }
  
  // Returns a unit triple in the direction (theta,phi), in radians.
  friend triple expi(double theta, double phi)
  {
    double sintheta=sin(theta);
    return triple(sintheta*cos(phi),sintheta*sin(phi),cos(theta));
  }
  
  friend istream& operator >> (istream& s, triple& z)
  {
    char c;
    s >> std::ws;
    bool paren=s.peek() == '('; // parenthesis are optional
    if(paren) s >> c;
    s >> z.x >> std::ws;
    if(s.peek() == ',') s >> c >> z.y;
    else z.y=0.0;
    if(s.peek() == ',') s >> c >> z.z;
    else z.z=0.0;
    if(paren) {
      s >> std::ws;
      if(s.peek() == ')') s >> c;
    }
    
    return s;
  }

  friend ostream& operator << (ostream& out, const triple& z)
  {
    out << "(" << z.x << "," << z.y << "," << z.z << ")";
    return out;
  }
  
};

triple expi(double theta, double phi);
  
struct node : public gc {
  triple pre,point,post;
public:
  node() {}
  node(const triple& pre, const triple& point, const triple& post)
    : pre(pre), point(point), post(post) {}
};
  
extern const unsigned maxdepth;

double cubiclength(const triple& z0, const triple& z0p, const triple& z1m,
		   const triple& z1, double goal=-1);
double bound(double *p, double (*m)(double, double), double b,
	     int depth=maxdepth);
double bound(triple *p, double (*m)(double, double), double (*f)(triple),
	     double b, int depth=maxdepth);
  
} //namespace camp

GC_DECLARE_PTRFREE(camp::triple);
GC_DECLARE_PTRFREE(camp::node);

#endif
