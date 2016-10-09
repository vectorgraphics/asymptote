/*****
 * angle.h
 * Andy Hammerlindl 2004/04/29
 *
 * For degree to radian conversion and visa versa.
 *****/

#ifndef ANGLE_H
#define ANGLE_H

#include <cmath>
#include "camperror.h"

namespace camp {

const double PI=acos(-1.0);

const double Cpiby180=PI/180.0;
const double C180bypi=180.0/PI;

inline double radians(double theta)
{
  return theta*Cpiby180;
}

inline double degrees(double theta)
{
  return theta*C180bypi;
}

// Wrapper for atan2 with sensible (lexical) argument order and (0,0) check
inline double angle(double x, double y)
{
  if(x == 0.0 && y == 0.0)
    reportError("taking angle of (0,0)");
  return atan2(y,x);
}

// Return an angle in the interval [0,360).
inline double principalBranch(double deg)
{
  deg = fmod(deg, 360);
  if(deg < 0)
    deg += 360;
  return deg;
}

} //namespace camp

#endif
