/* bound.cc
 * Bezier bounding-box computation.
 */

#include <cfloat>
#include <cmath>
#include "bound.h"
#include "triple.h"
#include "bbox.h"

namespace camp {

const double Fuzz2 = 1000.0 * DBL_EPSILON;
const double Fuzz = sqrt(Fuzz2);
const unsigned maxdepth = DBL_MANT_DIG;

} // namespace camp

namespace run {
    double norm(double *a, size_t n) {
        double result = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double v = a[i] < 0 ? -a[i] : a[i];
            if (v > result) result = v;
        }
        return result;
    }
}

namespace camp {

double cornerbound(double *P, double (*m)(double, double)) {
  double b=m(P[0],P[3]);
  b=m(b,P[12]);
  return m(b,P[15]);
}

double controlbound(double *P, double (*m)(double, double)) {
  double b=m(P[1],P[2]);
  b=m(b,P[4]);
  b=m(b,P[5]);
  b=m(b,P[6]);
  b=m(b,P[7]);
  b=m(b,P[8]);
  b=m(b,P[9]);
  b=m(b,P[10]);
  b=m(b,P[11]);
  b=m(b,P[13]);
  return m(b,P[14]);
}

double bound(double *P, double (*m)(double, double), double b,
             double fuzz, int depth) {
  b=m(b,cornerbound(P,m));
  if(m(-1.0,1.0)*(b-controlbound(P,m)) >= -fuzz || depth == 0)
    return b;

  --depth;
  fuzz *= 2;

  Split<double> c0(P[0],P[1],P[2],P[3]);
  Split<double> c1(P[4],P[5],P[6],P[7]);
  Split<double> c2(P[8],P[9],P[10],P[11]);
  Split<double> c3(P[12],P[13],P[14],P[15]);

  Split<double> c4(P[12],P[8],P[4],P[0]);
  Split<double> c5(c3.m0,c2.m0,c1.m0,c0.m0);
  Split<double> c6(c3.m3,c2.m3,c1.m3,c0.m3);
  Split<double> c7(c3.m5,c2.m5,c1.m5,c0.m5);
  Split<double> c8(c3.m4,c2.m4,c1.m4,c0.m4);
  Split<double> c9(c3.m2,c2.m2,c1.m2,c0.m2);
  Split<double> c10(P[15],P[11],P[7],P[3]);

  double s0[]={c4.m5,c5.m5,c6.m5,c7.m5,c4.m3,c5.m3,c6.m3,c7.m3,
               c4.m0,c5.m0,c6.m0,c7.m0,P[12],c3.m0,c3.m3,c3.m5};
  b=bound(s0,m,b,fuzz,depth);
  double s1[]={P[0],c0.m0,c0.m3,c0.m5,c4.m2,c5.m2,c6.m2,c7.m2,
               c4.m4,c5.m4,c6.m4,c7.m4,c4.m5,c5.m5,c6.m5,c7.m5};
  b=bound(s1,m,b,fuzz,depth);
  double s2[]={c0.m5,c0.m4,c0.m2,P[3],c7.m2,c8.m2,c9.m2,c10.m2,
               c7.m4,c8.m4,c9.m4,c10.m4,c7.m5,c8.m5,c9.m5,c10.m5};
  b=bound(s2,m,b,fuzz,depth);
  double s3[]={c7.m5,c8.m5,c9.m5,c10.m5,c7.m3,c8.m3,c9.m3,c10.m3,
               c7.m0,c8.m0,c9.m0,c10.m0,c3.m5,c3.m4,c3.m2,P[15]};
  return bound(s3,m,b,fuzz,depth);
}

double cornerboundtri(double *P, double (*m)(double, double)) {
  double b=m(P[0],P[6]);
  return m(b,P[9]);
}

double controlboundtri(double *P, double (*m)(double, double)) {
  double b=m(P[1],P[2]);
  b=m(b,P[3]);
  b=m(b,P[4]);
  b=m(b,P[5]);
  b=m(b,P[7]);
  return m(b,P[8]);
}

double boundtri(double *P, double (*m)(double, double), double b,
                double fuzz, int depth) {
  b=m(b,cornerboundtri(P,m));
  if(m(-1.0,1.0)*(b-controlboundtri(P,m)) >= -fuzz || depth == 0)
    return b;

  --depth;
  fuzz *= 2;

  Splittri<double> s(P);

  double l[]={s.l003,s.l102,s.l012,s.l201,s.l111,
              s.l021,s.l300,s.l210,s.l120,s.l030};
  b=boundtri(l,m,b,fuzz,depth);

  double r[]={s.l300,s.r102,s.r012,s.r201,s.r111,
              s.r021,s.r300,s.r210,s.r120,s.r030};
  b=boundtri(r,m,b,fuzz,depth);

  double u[]={s.l030,s.u102,s.u012,s.u201,s.u111,
              s.u021,s.r030,s.u210,s.u120,s.u030};
  b=boundtri(u,m,b,fuzz,depth);

  double c[]={s.r030,s.u201,s.r021,s.u102,s.c111,
              s.r012,s.l030,s.l120,s.l210,s.l300};
  return boundtri(c,m,b,fuzz,depth);
}

double ratiobound(triple z0, triple c0, triple c1, triple z1,
                  double (*m)(double, double),
                  double (*f)(const triple&)) {
  double MX=m(m(m(-z0.getx(),-c0.getx()),-c1.getx()),-z1.getx());
  double MY=m(m(m(-z0.gety(),-c0.gety()),-c1.gety()),-z1.gety());
  double Z=m(m(m(z0.getz(),c0.getz()),c1.getz()),z1.getz());
  double MZ=m(m(m(-z0.getz(),-c0.getz()),-c1.getz()),-z1.getz());
  return m(f(triple(-MX,-MY,Z)),f(triple(-MX,-MY,-MZ)));
}

double bound(triple z0, triple c0, triple c1, triple z1,
             double (*m)(double, double),
             double (*f)(const triple&), double b, double fuzz, int depth) {
  b=m(b,m(f(z0),f(z1)));
  if(m(-1.0,1.0)*(b-ratiobound(z0,c0,c1,z1,m,f)) >= -fuzz || depth == 0)
    return b;

  --depth;
  fuzz *= 2;

  triple m0=0.5*(z0+c0);
  triple m1=0.5*(c0+c1);
  triple m2=0.5*(c1+z1);
  triple m3=0.5*(m0+m1);
  triple m4=0.5*(m1+m2);
  triple m5=0.5*(m3+m4);

  b=bound(z0,m0,m3,m5,m,f,b,fuzz,depth);
  return bound(m5,m4,m2,z1,m,f,b,fuzz,depth);
}

} // namespace camp
