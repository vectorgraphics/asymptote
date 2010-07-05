import math;

int n=8, skip=3;

pair r(int k) { return unityroot(n,k); }

pen col=blue, col2=purple;

guide square=box((1,1),(-1,-1));

guide step(int mult)
{
  guide g;
  for(int k=0; k<n; ++k)
    g=g--r(mult*k);
  g=g--cycle;
  return g;
}

guide oct=step(1), star=step(skip);

guide wedge(pair z, pair v, real r, real a)
{
  pair w=expi(a/2.0);
  v=unit(v)*r;
  return shift(z)*((0,0)--v*w--v*conj(w)--cycle);
}

filldraw(square, col);
filldraw(oct, yellow);

// The interior angle of the points of the star.
real intang=pi*(1-((real)2skip)/((real)n));

for(int k=0; k<n; ++k) {
  pair z=midpoint(r(k)--r(k+1));
  guide g=wedge(z,-z,1,intang);
  filldraw(g,col2);
}

fill(star,yellow);
filldraw(star,evenodd+col);

size(5inch,0);
