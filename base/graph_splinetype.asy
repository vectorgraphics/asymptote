private import math;

typedef real[] splinetype(real[], real[]);

restricted real[] Spline(real[] x, real[] y);
restricted splinetype[] Spline;

string morepoints="interpolation requires at least 2 points";
string differentlengths="arrays have different lengths";
void checklengths(int x, int y, string text=differentlengths)
{
  if(x != y)
    abort(text+": "+string(x)+" != "+string(y));
}

void checkincreasing(real[] x) 
{
  if(!increasing(x,true))
    abort("strictly increasing array expected");
}

// Linear interpolation
real[] linear(real[] x, real[] y)
{
  int n=x.length;
  checklengths(n,y.length);
  real[] d=new real[n];
  for(int i=0; i < n-1; ++i)
    d[i]=(y[i+1]-y[i])/(x[i+1]-x[i]);
  d[n-1]=d[n-2];
  return d;
}

// Standard cubic spline interpolation with not-a-knot condition:
// s'''(x_2^-)=s'''(x_2^+) et s'''(x_(n_2)^-)=s'''(x_(n-2)^+)
// if n=2, linear interpolation is returned
// if n=3, an interpolation polynomial of degree <= 2 is returned:
// p(x_1)=y_1, p(x_2)=y_2, p(x_3)=y_3
real[] notaknot(real[] x, real[] y)
{
  int n=x.length;
  checklengths(n,y.length);
  checkincreasing(x);
  real[] d;
  if(n > 3) {
    real[] a=new real[n];
    real[] b=new real[n];
    real[] c=new real[n];
    real[] g=new real[n];
    b[0]=x[2]-x[1];
    c[0]=x[2]-x[0];
    a[0]=0;
    g[0]=((x[1]-x[0])^2*(y[2]-y[1])/b[0]+b[0]*(2*b[0]+3*(x[1]-x[0]))*
          (y[1]-y[0])/(x[1]-x[0]))/c[0];
    for(int i=1; i < n-1; ++i) {
      a[i]=x[i+1]-x[i];
      c[i]=x[i]-x[i-1];
      b[i]=2*(a[i]+c[i]);
      g[i]=3*(c[i]*(y[i+1]-y[i])/a[i]+a[i]*(y[i]-y[i-1])/c[i]);
    }
    c[n-1]=0;
    b[n-1]=x[n-2]-x[n-3];
    a[n-1]=x[n-1]-x[n-3];
    g[n-1]=((x[n-1]-x[n-2])^2*(y[n-2]-y[n-3])/b[n-1]+
            b[n-1]*(2*b[n-1]+3(x[n-1]-x[n-2]))*
            (y[n-1]-y[n-2])/(x[n-1]-x[n-2]))/a[n-1];
    d=tridiagonal(a,b,c,g);
  } else if(n == 2) {
    real val=(y[1]-y[0])/(x[1]-x[0]);
    d=new real[] {val,val};
  } else if(n == 3) {
    real a=(y[1]-y[0])/(x[1]-x[0]);
    real b=(y[2]-y[1])/(x[2]-x[1]);
    real c=(b-a)/(x[2]-x[0]);
    d=new real[] {a+c*(x[0]-x[1]),a+c*(x[1]-x[0]),a+c*(2*x[2]-x[0]-x[1])};
  } else abort(morepoints);
  return d;
}

// Standard cubic spline interpolation with periodic condition
// s'(a)=s'(b), s''(a)=s''(b), assuming that f(a)=f(b)
// if n=2, linear interpolation is returned
real[] periodic(real[] x, real[] y)
{
  int n=x.length;
  checklengths(n,y.length);
  checkincreasing(x);
  if(abs(y[n-1]-y[0]) > sqrtEpsilon*norm(y))
    abort("function values are not periodic");
  real[] d;
  if(n > 2) {
    real[] a=new real[n-1];
    real[] b=new real[n-1];
    real[] c=new real[n-1];
    real[] g=new real[n-1];
    c[0]=x[n-1]-x[n-2];
    a[0]=x[1]-x[0];
    b[0]=2*(a[0]+c[0]);
    g[0]=3*c[0]*(y[1]-y[0])/a[0]+3*a[0]*(y[n-1]-y[n-2])/c[0];
    for(int i=1; i < n-1; ++i) {
      a[i]=x[i+1]-x[i];
      c[i]=x[i]-x[i-1];
      b[i]=2*(a[i]+c[i]);
      g[i]=3*(c[i]*(y[i+1]-y[i])/a[i]+a[i]*(y[i]-y[i-1])/c[i]);
    }
    d=tridiagonal(a,b,c,g);
    d.push(d[0]);
  } else if(n == 2) {
    d=new real[] {0,0};
  } else abort(morepoints);
  return d;
}

// Standard cubic spline interpolation with the natural condition
// s''(a)=s''(b)=0.
// if n=2, linear interpolation is returned
// Don't use the natural type unless the underlying function 
// has zero second end points derivatives.
real[] natural(real[] x, real[] y)
{
  int n=x.length;
  checklengths(n,y.length);
  checkincreasing(x);
  real[] d;
  if(n > 2) {
    real[] a=new real[n];
    real[] b=new real[n];
    real[] c=new real[n];
    real[] g=new real[n];
    b[0]=2*(x[1]-x[0]);
    c[0]=x[1]-x[0];
    a[0]=0;
    g[0]=3*(y[1]-y[0]);
    for(int i=1; i < n-1; ++i) {
      a[i]=x[i+1]-x[i];
      c[i]=x[i]-x[i-1];
      b[i]=2*(a[i]+c[i]);
      g[i]=3*(c[i]*(y[i+1]-y[i])/a[i]+a[i]*(y[i]-y[i-1])/c[i]);
    }
    c[n-1]=0;
    a[n-1]=x[n-1]-x[n-2];
    b[n-1]=2*a[n-1];
    g[n-1]=3*(y[n-1]-y[n-2]);
    d=tridiagonal(a,b,c,g);
  } else if(n == 2) {
    real val=(y[1]-y[0])/(x[1]-x[0]);
    d=new real[] {val,val};
  } else abort(morepoints);
  return d;
}

// Standard cubic spline interpolation with clamped conditions f'(a), f'(b)
splinetype clamped(real slopea, real slopeb)
{
  return new real[] (real[] x, real[] y) {
    int n=x.length;
    checklengths(n,y.length);
    checkincreasing(x);
    real[] d;
    if(n > 2) {
      real[] a=new real[n];
      real[] b=new real[n];
      real[] c=new real[n];
      real[] g=new real[n];
      b[0]=x[1]-x[0];
      g[0]=b[0]*slopea;
      c[0]=0;
      a[0]=0;
      for(int i=1; i < n-1; ++i) {
        a[i]=x[i+1]-x[i];
        c[i]=x[i]-x[i-1];
        b[i]=2*(a[i]+c[i]);
        g[i]=3*(c[i]*(y[i+1]-y[i])/a[i]+a[i]*(y[i]-y[i-1])/c[i]);
      }
      c[n-1]=0;
      a[n-1]=0;
      b[n-1]=x[n-1]-x[n-2];
      g[n-1]=b[n-1]*slopeb;
      d=tridiagonal(a,b,c,g);
    } else if(n == 2) {
      d=new real[] {slopea,slopeb};
    } else abort(morepoints);
    return d;
  };
}

// Piecewise Cubic Hermite Interpolating Polynomial (PCHIP)
// Modified MATLAB code
// [1] Fritsch, F. N. and R. E. Carlson, 
//      "Monotone Piecewise Cubic Interpolation," 
//      SIAM J. Numerical Analysis, Vol. 17, 1980, pp.238-246.
// [2] Kahaner, David, Cleve Moler, Stephen Nash, 
//      Numerical Methods and Software, Prentice Hall, 1988.
real[] monotonic(real[] x, real[] y) 
{
  int n=x.length; 
  checklengths(n,y.length);
  checkincreasing(x);
  real[] d=new real[n]; 
  if(n > 2) {
    real[] h=new real[n-1];
    real[] del=new real[n-1];
    for(int i=0; i < n-1; ++i) {
      h[i]=x[i+1]-x[i]; 
      del[i]=(y[i+1]-y[i])/h[i]; 
    } 
    int j=0; 
    int k[]=new int[];
    for(int i=0; i < n-2; ++i)
      if((sgn(del[i])*sgn(del[i+1])) > 0) {k[j]=i; j=j+1;}

    real[] hs=new real[j];
    for(int i=0; i < j; ++i) hs[i]=h[k[i]]+h[k[i]+1];
    real w1[]=new real[j];
    real w2[]=new real[j];
    real dmax[]=new real[j];
    real dmin[]=new real[j];
    for(int i=0; i < j; ++i) {
      w1[i]=(h[k[i]]+hs[i])/(3*hs[i]);
      w2[i]=(h[k[i]+1]+hs[i])/(3*hs[i]);
      dmax[i]=max(abs(del[k[i]]),abs(del[k[i]+1]));
      dmin[i]=min(abs(del[k[i]]),abs(del[k[i]+1]));
    } 
    for(int i=0; i < n; ++i) d[i]=0;
    for(int i=0; i < j; ++i)
      d[k[i]+1]=dmin[i]/(w1[i]*(del[k[i]]/dmax[i])+w2[i]*(del[k[i]+1]/dmax[i])); 
    d[0]=((2*h[0]+h[1])*del[0]-h[0]*del[1])/(h[0]+h[1]);
    if(sgn(d[0]) != sgn(del[0])) {d[0]=0;}
    else if((sgn(del[0]) != sgn(del[1])) && (abs(d[0]) > abs(3*del[0])))
      d[0]=3*del[0];

    d[n-1]=((2*h[n-2]+h[n-3])*del[n-2]-h[n-2]*del[n-2])/(h[n-2]+h[n-3]);
    if(sgn(d[n-1]) != sgn(del[n-2])) {d[n-1]=0;}
    else if((sgn(del[n-2]) != sgn(del[n-3])) &&
            (abs(d[n-1]) > abs(3*del[n-2])))
      d[n-1]=3*del[n-2];
  } else if(n == 2) {
    d[0]=d[1]=(y[1]-y[0])/(x[1]-x[0]);
  } else abort(morepoints);
  return d;
} 

// Return standard cubic spline interpolation as a guide
guide hermite(real[] x, real[] y, splinetype splinetype=null)
{
  int n=x.length;
  if(n == 0) return nullpath;

  guide g=(x[0],y[0]);
  if(n == 1) return g;
  if(n == 2) return g--(x[1],y[1]);

  if(splinetype == null)
    splinetype=(x[0] == x[x.length-1] && y[0] == y[y.length-1]) ?
      periodic : notaknot;

  real[] dy=splinetype(x,y);
  for(int i=1; i < n; ++i) {
    pair z=(x[i],y[i]);
    real dx=x[i]-x[i-1];
    g=g..controls((x[i-1],y[i-1])+dx*(1,dy[i-1])/3) and (z-dx*(1,dy[i])/3)..z;
  }
  return g;
}
