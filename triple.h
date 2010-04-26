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

typedef double Triple[3];
  
class triple : virtual public gc {
  double x;
  double y;
  double z;

public:
  triple() : x(0.0), y(0.0), z(0.0) {}
  triple(double x, double y=0.0, double z=0.0) : x(x), y(y), z(z) {}
  triple(const Triple& v) : x(v[0]), y(v[1]), z(v[2]) {}

  virtual ~triple() {}
  
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

  friend triple operator* (double s, const triple& z)
  {
    return triple(s*z.x, s*z.y, s*z.z);
  }

  friend triple operator* (const triple& z, double s)
  {
    return triple(z.x*s, z.y*s, z.z*s);
  }

  friend triple operator/ (const triple& z, double s)
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

  const triple& operator-= (const triple& w)
  {
    x -= w.x;
    y -= w.y;
    z -= w.z;
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
  
  friend double length(const triple& v)
  {
    return v.length();
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
  
  friend triple unit(const triple& v)
  {
    double scale=v.length();
    if(scale != 0.0) scale=1.0/scale;
    return triple(v.x*scale,v.y*scale,v.z*scale);
  }
  
  friend double dot(const triple& u, const triple& v)
  {
    return u.x*v.x+u.y*v.y+u.z*v.z;
  }

  friend triple cross(const triple& u, const triple& v) 
  {
    return triple(u.y*v.z-u.z*v.y,
                  u.z*v.x-u.x*v.z,
                  u.x*v.y-v.x*u.y);
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
  
// Return the component of vector v perpendicular to a unit vector u.
inline triple perp(triple v, triple u)
{
  return v-dot(v,u)*u;
}

} //namespace camp

GC_DECLARE_PTRFREE(camp::triple);

#endif
