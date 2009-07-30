// Lagrange and Hermite interpolation in Asymptote
// Author: Olivier Guibé
// Acknowledgements: Philippe Ivaldi

// diffdiv(x,y) computes Newton's Divided Difference for
// Lagrange interpolation with distinct values {x_0,..,x_n} in the array x
// and values y_0,...,y_n in the array y,

// hdiffdiv(x,y,dyp) computes Newton's Divided Difference for
// Hermite interpolation where dyp={dy_0,...,dy_n}. 
// 
// fhorner(x,coeff) uses Horner's rule to compute the polynomial
// a_0+a_1(x-x_0)+a_2(x-x_0)(x-x_1)+...+a_n(x-x_0)..(x-x_{n-1}),
// where coeff={a_0,a_1,...,a_n}.

// fspline does standard cubic spline interpolation of a function f
// on the interval [a,b].
// The points a=x_1 < x_2 < .. < x_n=b form the array x;
// the points y_1=f(x_1),....,y_n=f(x_n) form the array y
// We use the Hermite form for the spline.

// The syntax is:
// s=fspline(x,y);                            default not_a_knot condition
// s=fspline(x,y,natural);                    natural spline
// s=fspline(x,y,periodic);                   periodic spline
// s=fspline(x,y,clamped(1,1));               clamped spline
// s=fspline(x,y,monotonic);                  piecewise monotonic spline

// Here s is a real function that is constant on (-infinity,a] and [b,infinity).

private import math;
import graph_splinetype;

typedef real fhorner(real);

struct horner {
  // x={x0,..,xn}(not necessarily distinct)
  // a={a0,..,an} corresponds to the polyonmial
  // a_0+a_1(x-x_0)+a_2(x-x_0)(x-x_1)+...+a_n(x-x_0)..(x-x_{n-1}),
  real[] x;
  real[] a;
}

// Evaluate p(x)=d0+(x-x0)(d1+(x-x1)+...+(d(n-1)+(x-x(n-1))*dn)))
// via Horner's rule: n-1 multiplications, 2n-2 additions.
fhorner fhorner(horner sh)
{
  int n=sh.x.length;
  checklengths(n,sh.a.length);
  return new real(real x) {
    real s=sh.a[n-1];
    for(int k=n-2; k >= 0; --k)
      s=sh.a[k]+(x-sh.x[k])*s;
    return s;
  };
}
      
// Newton's Divided Difference method: n(n-1)/2 divisions, n(n-1) additions.
horner diffdiv(real[] x, real[] y)
{
  int n=x.length;
  horner s;
  checklengths(n,y.length);
  for(int i=0; i < n; ++i)
    s.a[i]=y[i];
  for(int k=0; k < n-1; ++k) {
    for(int i=n-1; i > k; --i) {
      s.a[i]=(s.a[i]-s.a[i-1])/(x[i]-x[i-k-1]);
    }
  }
  s.x=x;
  return s;
}

// Newton's Divided Difference for simple Hermite interpolation,
// where one specifies both p(x_i) and p'(x_i).
horner hdiffdiv(real[] x, real[] y, real[] dy)
{
  int n=x.length;
  horner s;
  checklengths(n,y.length);
  checklengths(n,dy.length);
  for(int i=0; i < n; ++i) {
    s.a[2*i]=y[i];
    s.a[2*i+1]=dy[i];
    s.x[2*i]=x[i];
    s.x[2*i+1]=x[i];
  }

  for(int i=n-1; i > 0; --i)
    s.a[2*i]=(s.a[2*i]-s.a[2*i-2])/(x[i]-x[i-1]);

  int stop=2*n-1;
  for(int k=1; k < stop; ++k) {
    for(int i=stop; i > k; --i) {
      s.a[i]=(s.a[i]-s.a[i-1])/(s.x[i]-s.x[i-k-1]);
    }
  }
  return s;
}

typedef real realfunction(real);

// piecewise Hermite interpolation:
// return the piecewise polynomial p(x), where on [x_i,x_i+1], deg(p) <= 3,
// p(x_i)=y_i, p(x_{i+1})=y_i+1, p'(x_i)=dy_i, and p'(x_{i+1})=dy_i+1.
// Outside [x_1,x_n] the returned function is constant: y_1 on (infinity,x_1]
// and y_n on [x_n,infinity).
realfunction pwhermite(real[] x, real[] y, real[] dy)
{
  int n=x.length;
  checklengths(n,y.length);
  checklengths(n,dy.length);
  if(n < 2) abort(morepoints);
  if(!increasing(x,strict=true)) abort("array x is not strictly increasing");
  return new real(real t) {
    int i=search(x,t);
    if(i == n-1) {
      i=n-2;
      t=x[n-1];
    } else if(i == -1) {
      i=0;
      t=x[0];
    }
    real h=x[i+1]-x[i];
    real delta=(y[i+1]-y[i])/h;
    real e=(3*delta-2*dy[i]-dy[i+1])/h;
    real f=(dy[i]-2*delta+dy[i+1])/h^2;
    real s=t-x[i];
    return y[i]+s*(dy[i]+s*(e+s*f));
  };
}

realfunction fspline(real[] x, real[] y, splinetype splinetype=notaknot)
{
  real[] dy=splinetype(x,y);
  return new real(real t) {
    return pwhermite(x,y,dy)(t);
  }; 
}
