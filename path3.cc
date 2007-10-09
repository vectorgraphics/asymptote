/*****
 * path3.cc
 * John Bowman
 *
 * Compute information for a three-dimensional path.
 *****/

#include <cfloat>

#include "path.h"
#include "triple.h"

namespace camp {

// Calculate coefficients of Bezier derivative.
static inline void derivative(triple& a, triple& b, triple& c,
			      triple z0, triple z0p, triple z1m, triple z1)
{
  a=z1-z0+3.0*(z0p-z1m);
  b=2.0*(z0+z1m)-4.0*z0p;
  c=z0p-z0;
}

static triple a,b,c;

static double ds(double t)
{
  double dx=quadratic(a.getx(),b.getx(),c.getx(),t);
  double dy=quadratic(a.gety(),b.gety(),c.gety(),t);
  double dz=quadratic(a.getz(),b.getz(),c.getz(),t);
  return sqrt(dx*dx+dy*dy+dz*dz);
}

// Calculates arclength of a cubic using adaptive simpson integration.
double cubiclength(const triple& z0, const triple& z0p, const triple& z1m,
		   const triple& z1, double goal)
{
  double L,integral;
  derivative(a,b,c,z0,z0p,z1m,z1);
  
  if(!simpson(integral,ds,0.0,1.0,DBL_EPSILON,1.0))
    reportError("nesting capacity exceeded in computing arclength");
  L=3.0*integral;
  if(goal < 0 || goal >= L) return L;
  
  static const double third=1.0/3.0;
  double t=goal/L;
  goal *= third;
  if(!unsimpson(goal,ds,0.0,t,10.0*DBL_EPSILON,integral,1.0,sqrt(DBL_EPSILON)))
    reportError("nesting capacity exceeded in computing arctime");
  return -t;
}

} //namespace camp
  
