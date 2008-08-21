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

struct split {
  double m0,m1,m2,m3,m4,m5;
  split(double z0, double c0, double c1, double z1) {
    m0=0.5*(z0+c0);
    m1=0.5*(c0+c1);
    m2=0.5*(c1+z1);
    m3=0.5*(m0+m1);
    m4=0.5*(m1+m2);
    m5=0.5*(m3+m4);
  }
};
  
struct split3 {
  triple m0,m1,m2,m3,m4,m5;
  split3(triple z0, triple c0, triple c1, triple z1) {
    m0=0.5*(z0+c0);
    m1=0.5*(c0+c1);
    m2=0.5*(c1+z1);
    m3=0.5*(m0+m1);
    m4=0.5*(m1+m2);
    m5=0.5*(m3+m4);
  }
};
  
double cornerbound(double *p, double (*m)(double, double)) 
{
  double b=m(p[0],p[3]);
  b=m(b,p[12]);
  return m(b,p[15]);
}

double controlbound(double *p, double (*m)(double, double)) 
{
  double b=m(p[1],p[2]);
  b=m(b,p[4]);
  b=m(b,p[5]);
  b=m(b,p[6]);
  b=m(b,p[7]);
  b=m(b,p[8]);
  b=m(b,p[9]);
  b=m(b,p[10]);
  b=m(b,p[11]);
  b=m(b,p[13]);
  return m(b,p[14]);
}

double cornerbound(triple *p, double (*m)(double, double), double (*f)(triple)) 
{
  double b=m(f(p[0]),f(p[3]));
  b=m(b,f(p[12]));
  return m(b,f(p[15]));
}

double controlbound(triple *p, double (*m)(double, double),
		    double (*f)(triple)) 
{
  double b=m(f(p[1]),f(p[2]));
  b=m(b,f(p[4]));
  b=m(b,f(p[5]));
  b=m(b,f(p[6]));
  b=m(b,f(p[7]));
  b=m(b,f(p[8]));
  b=m(b,f(p[9]));
  b=m(b,f(p[10]));
  b=m(b,f(p[11]));
  b=m(b,f(p[13]));
  return m(b,f(p[14]));
}

double bound(double *p, double (*m)(double, double), double b, int depth)
{
  b=m(b,cornerbound(p,m));
  if(m(-1.0,1.0)*(b-controlbound(p,m)) >= -sqrtFuzz || depth == 0)
    return b;
  --depth;

  split c0(p[0],p[1],p[2],p[3]);
  split c1(p[4],p[5],p[6],p[7]);
  split c2(p[8],p[9],p[10],p[11]);
  split c3(p[12],p[13],p[14],p[15]);

  split c4(p[12],p[8],p[4],p[0]);
  split c5(c3.m0,c2.m0,c1.m0,c0.m0);
  split c6(c3.m3,c2.m3,c1.m3,c0.m3);
  split c7(c3.m5,c2.m5,c1.m5,c0.m5);
  split c8(c3.m4,c2.m4,c1.m4,c0.m4);
  split c9(c3.m5,c2.m5,c1.m5,c0.m5);
  split c10(p[15],p[11],p[7],p[3]);

  // Check all 4 Bezier subpatches.
  double s0[]={c4.m5,c5.m5,c6.m5,c7.m5,c4.m3,c5.m3,c6.m3,c7.m3,
                 c4.m0,c5.m0,c6.m0,c7.m0,p[12],c3.m0,c3.m3,c3.m5};
  b=bound(s0,m,b,depth);
  double s1[]={p[0],c0.m0,c0.m3,c0.m5,c4.m2,c5.m2,c6.m2,c7.m2,
                 c4.m4,c5.m4,c6.m4,c7.m4,c4.m5,c5.m5,c6.m5,c7.m5};
  b=bound(s1,m,b,depth);
  double s2[]={c0.m5,c0.m4,c0.m2,p[3],c7.m2,c8.m2,c9.m2,c10.m2,
                 c7.m4,c8.m4,c9.m4,c10.m4,c7.m5,c8.m5,c9.m5,c10.m5};
  b=bound(s2,m,b,depth);
  double s3[]={c7.m5,c8.m5,c9.m5,c10.m5,c7.m3,c8.m3,c9.m3,c10.m3,
                 c7.m0,c8.m0,c9.m0,c10.m0,c3.m5,c3.m4,c3.m2,p[15]};
  return bound(s3,m,b,depth);
}
  
double bound(triple *p, double (*m)(double, double), double (*f)(triple),
	     double b, int depth)
{
  b=m(b,cornerbound(p,m,f));
  if(m(-1.0,1.0)*(b-controlbound(p,m,f)) >= -sqrtFuzz || depth == 0)
    return b;
  --depth;

  split3 c0(p[0],p[1],p[2],p[3]);
  split3 c1(p[4],p[5],p[6],p[7]);
  split3 c2(p[8],p[9],p[10],p[11]);
  split3 c3(p[12],p[13],p[14],p[15]);

  split3 c4(p[12],p[8],p[4],p[0]);
  split3 c5(c3.m0,c2.m0,c1.m0,c0.m0);
  split3 c6(c3.m3,c2.m3,c1.m3,c0.m3);
  split3 c7(c3.m5,c2.m5,c1.m5,c0.m5);

  split3 c8(c3.m4,c2.m4,c1.m4,c0.m4);
  split3 c9(c3.m5,c2.m5,c1.m5,c0.m5);
  split3 c10(p[15],p[11],p[7],p[3]);

  // Check all 4 Bezier subpatches.
  
  triple s0[]={c4.m5,c5.m5,c6.m5,c7.m5,c4.m3,c5.m3,c6.m3,c7.m3,
	       c4.m0,c5.m0,c6.m0,c7.m0,p[12],c3.m0,c3.m3,c3.m5};
  b=bound(s0,m,f,b,depth);
  triple s1[]={p[0],c0.m0,c0.m3,c0.m5,c4.m2,c5.m2,c6.m2,c7.m2,
	       c4.m4,c5.m4,c6.m4,c7.m4,c4.m5,c5.m5,c6.m5,c7.m5};
  b=bound(s1,m,f,b,depth);
  triple s2[]={c0.m5,c0.m4,c0.m2,p[3],c7.m2,c8.m2,c9.m2,c10.m2,
	       c7.m4,c8.m4,c9.m4,c10.m4,c7.m5,c8.m5,c9.m5,c10.m5};
  b=bound(s2,m,f,b,depth);
  triple s3[]={c7.m5,c8.m5,c9.m5,c10.m5,c7.m3,c8.m3,c9.m3,c10.m3,
                 c7.m0,c8.m0,c9.m0,c10.m0,c3.m5,c3.m4,c3.m2,p[15]};
  return bound(s3,m,f,b,depth);
}

} //namespace camp
  
