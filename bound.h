/* bound.h
 * Bezier bounding-box computation.
 * Provides tight bounds for Bezier patches and triangles via recursive
 * de Casteljau subdivision.
 */

#pragma once

#include <cstddef>

namespace camp {
class triple;

// De Casteljau subdivision helper for cubic Bezier curves/patches.
template<class T>
struct Split {
  T m0,m1,m2,m3,m4,m5;
  Split(T z0, T c0, T c1, T z1) {
    m0=0.5*(z0+c0);
    m1=0.5*(c0+c1);
    m2=0.5*(c1+z1);
    m3=0.5*(m0+m1);
    m4=0.5*(m1+m2);
    m5=0.5*(m3+m4);
  }
};

// De Casteljau subdivision helper for quadratic Bezier triangles.
template<class T>
struct Splittri {
  T l003,p102,p012,p201,p111,p021,r300,p210,p120,u030;
  T u021,u120;
  T p033,p231,p330;
  T p123;
  T l012,p312,r210,l102,p303,r201;
  T u012,u210,l021,p4xx,r120,px4x,pxx4,l201,r102;
  T l210,r012,l300;
  T r021,u201,r030;
  T u102,l120,l030;
  T l111,r111,u111,c111;

  Splittri(const T *p) {
    l003=p[0]; p102=p[1]; p012=p[2]; p201=p[3]; p111=p[4];
    p021=p[5]; r300=p[6]; p210=p[7]; p120=p[8]; u030=p[9];

    u021=0.5*(u030+p021); u120=0.5*(u030+p120);
    p033=0.5*(p021+p012); p231=0.5*(p120+p111); p330=0.5*(p120+p210);
    p123=0.5*(p012+p111);

    l012=0.5*(p012+l003); p312=0.5*(p111+p201); r210=0.5*(p210+r300);
    l102=0.5*(l003+p102); p303=0.5*(p102+p201); r201=0.5*(p201+r300);

    u012=0.5*(u021+p033); u210=0.5*(u120+p330);
    l021=0.5*(p033+l012); p4xx=0.5*p231+0.25*(p111+p102);
    r120=0.5*(p330+r210); px4x=0.5*p123+0.25*(p111+p210);
    pxx4=0.25*(p021+p111)+0.5*p312;
    l201=0.5*(l102+p303); r102=0.5*(p303+r201);

    l210=0.5*(px4x+l201); r012=0.5*(px4x+r102); l300=0.5*(l201+r102);
    r021=0.5*(pxx4+r120); u201=0.5*(u210+pxx4); r030=0.5*(u210+r120);
    u102=0.5*(u012+p4xx); l120=0.5*(l021+p4xx); l030=0.5*(u012+l021);

    l111=0.5*(p123+l102); r111=0.5*(p312+r210);
    u111=0.5*(u021+p231); c111=0.25*(p033+p330+p303+p111);
  }
};

// Recursive subdivision bound functions (from path3.cc).
double bound(double *P, double (*m)(double, double), double b, double fuzz, int depth);
double boundtri(double *P, double (*m)(double, double), double b, double fuzz, int depth);

// Cubic Bezier curve bound
double ratiobound(triple z0, triple c0, triple c1, triple z1,
                        double (*m)(double, double),
                        double (*f)(const triple&));
double bound(triple z0, triple c0, triple c1, triple z1,
             double (*m)(double, double),
             double (*f)(const triple&), double b, double fuzz, int depth);

// Fuzz constants (from path.cc).
extern const double Fuzz2;
extern const double Fuzz;
extern const unsigned maxdepth;
}

namespace run {
    // L-infinity norm of an array
    double norm(double *a, size_t n);
}
