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
// s=fspline(x,y,clamped(1,1));               clamped spline
// s=fspline(x,y,periodic);                   periodic spline
// s=fspline(x,y,natural);                    natural spline
//
// Here s is a real function that is constant on (-infinity,a] and [b,infinity).

typedef real fhorner(real);

struct horner {
  // x={x0,..,xn}(not necessarily distinct)
  // a={a0,..,an} corresponds to the polyonmial
  // a_0+a_1(x-x_0)+a_2(x-x_0)(x-x_1)+...+a_n(x-x_0)..(x-x_{n-1}),
  real[] x;
  real[] a;
}

private string differentlengths="arrays have different lengths";

fhorner fhorner(horner sh)
{// Evaluate p(x)=d0+(x-x0)(d1+(x-x1).....+(d(n-1)+(x-x(n-1))*dn)))
 // via Horner's rule: n-1 multiplications, 2n-2 additions.
  return new real(real x) {
    int n=sh.x.length;
    if(n != sh.a.length) abort(differentlengths);
    real s;
    s=sh.a[n-1];
    for(int k=n-2; k >= 0; --k)
      s=sh.a[k]+(x-sh.x[k])*s;
    return s;
  };
}
      
horner diffdiv(real[] x, real[] y)
{// Newton's Divided Difference method: n(n-1)/2 divisions, n(n-1) additions.
  int n=x.length;
  horner s;
  if(n != y.length) abort(differentlengths);
  real[] d;
  for(int i=0; i < n; ++i) {
    s.a[i]=y[i];
  }
  for(int k=0; k < n-1; ++k) {
    for(int i=n-1; i > k; --i) {
      s.a[i]=(s.a[i]-s.a[i-1])/(x[i]-x[i-k-1]);
    }
  }
  s.x=x;
  return s;
}

horner hdiffdiv(real[] x, real[] y, real[] dy)
{ // Newton's Divided Difference for simple Hermite interpolation,
  // where one specifies both p(x_i) and p'(x_i).
  int n=x.length;
  horner s;
  if(n != y.length || n != dy.length) abort(differentlengths);
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
typedef real[] splinetype(real[], real[]);

private string onepoint="interpolation requires at least 2 points";

realfunction pwhermite(real[] x, real[] y, real[] dy)
{
  // piecewise Hermite interpolation:
  // return the piecewise polynomial p(x), where on [x_i,x_i+1], deg(p) <= 3,
  // p(x_i)=y_i, p(x_{i+1})=y_i+1, p'(x_i)=dy_i, and p'(x_{i+1})=dy_i+1.
  // Outside [x_1,x_n] the returned function is constant: y_1 on (infinity,x_1]
  // and y_n on [x_n,infinity).
  if(x.length < 2) abort(onepoint);
  if(x.length != y.length) abort(differentlengths);
  if(x.length != dy.length) abort(differentlengths);
  return new real(real t) {
    int n=x.length;
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
    return(y[i]+s*(dy[i]+s*(e+s*f)));
  };
}

splinetype notaknot()
{  // Standard cubic spline interpolation with not a knot condition:
  // s'''(x_2^-)=s'''(x_2^+) et s'''(x_(n_2)^-)=s'''(x_(n-2)^+)
  // if n=2, linear interpolation is returned
  // if n=3, an interpolation polynomial of degree <= 2 is returned:
  // p(x_1)=y_1, p(x_2)=y_2, p(x_3)=y_3
  return new real[] (real[] x, real[] y) {
    int n=x.length;
    real[] d;
    if(n < 2) abort(onepoint);
    if(n != y.length) abort(differentlengths);
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
    }
    if(n == 2) {
      d[0]=(y[1]-y[0])/(x[1]-x[0]);
      d[1]=d[0];
    }
    if(n == 3) {
      real a=(y[1]-y[0])/(x[1]-x[0]);
      real b=(y[2]-y[1])/(x[2]-x[1]);
      real c=(b-a)/(x[2]-x[0]);
      d[0]=a+c*(x[0]-x[1]);
      d[1]=a+c*(x[1]-x[0]);
      d[2]=a+c*(2*x[2]-x[0]-x[1]);
    }
    return d;
  };
}

splinetype notaknot=notaknot();

splinetype periodic()
{
  // Standard cubic spline interpolation with periodic condition
  // s'(a)=s'(b), s''(a)=s''(b), assuming that f(a)=f(b)
  // if n=2, linear interpolation is returned

  return new real[] (real[] x, real[] y){
    int n=x.length;
    real[] d;
    if(n < 2) abort(onepoint);
    if(n != y.length) abort(differentlengths);
    if(y[n-1] != y[0]) abort("Function values are not periodic");
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
    };
    if(n == 2) {
      d[0]=0;
      d[1]=0;
    }
    return d;
  };
}

splinetype periodic=periodic();


splinetype natural()
{ // Standard cubic spline interpolation with the natural condition
  // s''(a)=s''(b)=0.
  // if n=2, linear interpolation is returned
  // Don't use the natural type unless the underlying function 
  // has zero second end points derivatives.
 
  return new real[] (real[] x, real[] y){
    int n=x.length;
    real[] d;
    if(n < 2) abort(onepoint);
    if(n != y.length) abort(differentlengths);
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
    }
    if(n == 2) {
      d[0]=(y[1]-y[0])/(x[1]-x[0]);
      d[1]=d[0];
    }
    return d;
  };
}

splinetype natural=natural();

splinetype clamped(real slopea, real slopeb)
{
  // Standard cubic spline interpolation with clamped conditions f'(a), f'(b)
  real[] de={slopea,slopeb};

  return new real[] (real[] x, real[] y) {
    int n=x.length;
    real[] d;
    if(n < 2) abort(onepoint);
    if(n != y.length) abort(differentlengths);
    if(n > 2) {
      real[] a=new real[n];
      real[] b=new real[n];
      real[] c=new real[n];
      real[] g=new real[n];
      b[0]=x[1]-x[0];
      g[0]=b[0]*de[0];
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
      g[n-1]=b[n-1]*de[1];
      d=tridiagonal(a,b,c,g);
    }
    if(n == 2) {
      d[0]=de[0];
      d[1]=de[1];
    }
    return d;
  };
}

realfunction fspline(real[] x, real[] y, splinetype splinetype=notaknot)
{
  real[] dy=splinetype(x,y);
  return new real(real t) {
    return(pwhermite(x,y,dy)(t));
  }; 
}
