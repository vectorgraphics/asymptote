/*****
 * angle.h
 * Andy Hammerlindl 2004/04/29
 *
 * For degree to radian conversion and visa versa.
 *****/

#ifndef ANGLE_H
#define ANGLE_H

#include <cmath>

namespace camp {

const double PI=acos(-1.0);

const double Cpiby180=PI/180.0;
const double C180bypi=180.0/PI;
  
inline double radians(const double theta)
{
  return theta*Cpiby180;
}

inline double degrees(const double theta)
{
  return theta*C180bypi;
}

} //namespace camp

#endif
