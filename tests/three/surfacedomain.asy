import TestLib;
import graph3;

bool close(triple a, triple b)
{
  real norm=max(abs(a),abs(b),1);
  return abs(a-b) <= 1e-9*norm;
}

// The evaluations that abort are tested in tests/test_surface_errors.py.

StartTest("surface domain: boundary and rounding");
{
  triple f(pair z) {return (z.x,z.y,z.x*z.y);}
  pair a=(1,2), b=(5,8);
  int nu=4, nv=3;
  surface s=surface(f,a,b,nu,nv);
  assert(!s.index.cyclic && !s.vcyclic);

  // Both ends of the domain belong to it.
  assert(close(s.point(0,0),f(a)));
  assert(close(s.point(nu,nv),f(b)));
  assert(close(s.point(nu,1.5),f((b.x,5))));
  assert(close(s.point(2.5,nv),f((3.5,b.y))));
  assert(close(s.paramPoint(a.x,a.y),f(a)));
  assert(close(s.paramPoint(b.x,b.y),f(b)));
  assert(close(s.paramPoint(2,b.y),f((2,b.y))));
  assert(close(s.normal(nu,nv),s.normal(nu-1e-12,nv-1e-12)));

  // A coordinate that misses the domain only by rounding is moved to the
  // nearest boundary rather than extrapolated or rejected.
  real tiny=1e-10;
  assert(tiny < sqrtEpsilon);
  assert(s.point(-tiny,1) == s.point(0,1));
  assert(s.point(nu+tiny,1) == s.point(nu,1));
  assert(s.point(1,-tiny) == s.point(1,0));
  assert(s.point(1,nv+tiny) == s.point(1,nv));
  assert(s.normal(nu+tiny,nv+tiny) == s.normal(nu,nv));
  assert(s.paramPoint(a.x-tiny,3) == s.paramPoint(a.x,3));
  assert(s.paramPoint(b.x+tiny,3) == s.paramPoint(b.x,3));
  assert(s.paramNormal(2,b.y+tiny) == s.paramNormal(2,b.y));

  // A transformed surface or a copy has the same domain.
  transform3 T=shift(1,2,3)*rotate(90,Z);
  assert(close((T*s).paramPoint(b.x,b.y),T*f(b)));
  assert(close(surface(s).paramPoint(b.x,b.y),f(b)));
}
EndTest();

StartTest("surface domain: cyclic directions");
{
  // A torus is cyclic in both directions.
  path3 g=shift(3X)*rotate(90,X)*unitcircle3;
  int n=6;
  surface s=surface(O,g,Z,n);
  int nv=length(g);
  assert(s.index.cyclic && s.vcyclic);

  triple p=s.point(1.25,0.5);
  assert(close(s.point(1.25+n,0.5),p));
  assert(close(s.point(1.25-n,0.5),p));
  assert(close(s.point(1.25+3n,0.5-2nv),p));
  assert(close(s.point(1.25,0.5+nv),p));
  assert(close(s.point(1.25,0.5-nv),p));
  assert(close(s.point(-0.5,-0.25),s.point(n-0.5,nv-0.25)));
  assert(close(s.point(n,nv),s.point(0,0)));
  assert(close(s.normal(1.25+n,0.5-nv),s.normal(1.25,0.5)));

  // The parametric coordinate u is the angle of rotation in radians.
  assert(close(s.paramPoint(pi/3,0.5),rotate(60,Z)*point(g,0.5)));
  triple p=s.paramPoint(1,0.5);
  assert(close(s.paramPoint(1+2pi,0.5),p));
  assert(close(s.paramPoint(1-2pi,0.5+nv),p));
  assert(close(s.paramPoint(-1,-0.5),s.paramPoint(2pi-1,nv-0.5)));
  assert(close(s.paramNormal(1+2pi,0.5-nv),s.paramNormal(1,0.5)));
}
EndTest();

StartTest("surface domain: mixed cyclic and noncyclic");
{
  // A cylinder is cyclic around its axis but not along it.
  path3 g=(1,0,0)--(1,0,1)--(1,0,3);
  int n=4;
  surface s=surface(O,g,Z,n);
  assert(s.index.cyclic && !s.vcyclic);

  assert(close(s.point(0.5+n,1.5),s.point(0.5,1.5)));
  assert(close(s.point(-0.5,2),s.point(n-0.5,2)));
  // The noncyclic direction still ends at its boundary.
  assert(close(s.point(0,2),(1,0,3)));
  assert(s.point(0.5+n,2+1e-10) == s.point(0.5+n,2));
  assert(close(s.paramPoint(5pi/2,2),(0,1,3)));

  // A partial rotation is not cyclic in either direction.
  surface s=surface(O,g,Z,n,0,90);
  assert(!s.index.cyclic && !s.vcyclic);
  assert(close(s.point(n,2),(0,1,3)));
  assert(close(s.paramPoint(pi/2,2),(0,1,3)));
}
EndTest();
