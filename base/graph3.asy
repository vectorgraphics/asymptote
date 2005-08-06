// Three-dimensional graphing routines

import math;
import graph;
import three;

// Under construction.

typedef guide3 graph(triple F(real), real, real, int);

public graph graph(guide3 join(... guide3[]))
{
  return new guide3(triple F(real), real a, real b, int n) {
    guide3 g;
    real width=n == 0 ? 0 : (b-a)/n;
    for(int i=0; i <= n; ++i) {
      real x=a+width*i;
      g=join(g,F(x));	
    }	
    return g;
  };
}

public guide3 Straight(... guide3[])=operator --;
		       
triple Scale(picture pic, triple v)
{
  return (pic.scale.x.T(v.x),pic.scale.y.T(v.y),pic.scale.z.T(v.z));
}

typedef guide3 interpolate(... guide3[]);

guide3 graph(picture pic=currentpicture, real x(real), real y(real),
	     real z(real), real a, real b, int n=ngraph,
	     interpolate join=operator --)
{
  return graph(join)(new triple (real t) {return Scale(pic,(x(t),y(t),z(t)));},
		     a,b,n);
}
