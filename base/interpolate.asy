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

typedef real fhorner(real);

struct horner {
  // x={x0,..,xn}(not necessarily distinct)
  // a={a0,..,an} corresponds to the polyonmial
  // a_0+a_1(x-x_0)+a_2(x-x_0)(x-x_1)+...+a_n(x-x_0)..(x-x_{n-1}),
  real[] x;
  real[] a;
}

string differentlengths="arrays have different lengths";

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
  real[] d;
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
