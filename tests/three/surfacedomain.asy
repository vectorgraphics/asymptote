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

StartTest("surface domain: domain and paramCoords");
{
  triple f(pair z) {return (z.x,z.y,z.x*z.y);}
  pair a=(1,2), b=(5,8);
  int nu=4, nv=3;
  surface s=surface(f,a,b,nu,nv);

  // paramCoords maps surface coordinates to parametric coordinates.
  assert(s.paramCoords(0,0) == a);
  assert(abs(s.paramCoords(nu,nv)-b) <= 1e-12);
  assert(abs(s.paramCoords(1,1.5)-(2,5)) <= 1e-12);
  pair z=s.paramCoords(2.5,0.75);
  assert(close(s.paramPoint(z.x,z.y),s.point(2.5,0.75)));

  // A surface built from arrays has surface coordinates until it is given a
  // domain.
  surface t=surface(f,uniform(a.x,b.x,nu),uniform(a.y,b.y,nv),Spline);
  assert(t.paramCoords(1,2) == (1,2));
  assert(close(t.paramPoint(nu,nv),f(b)));
  t.domain(a,b);
  assert(t.paramCoords(0,0) == a);
  assert(close(t.paramPoint(b.x,b.y),f(b)));
  assert(close(t.paramPoint(2,5),s.paramPoint(2,5)));

  // Whichever corner comes first corresponds to the surface coordinates (0,0).
  t.domain(b,a);
  assert(t.paramCoords(0,0) == b);
  assert(close(t.paramPoint(b.x,b.y),f(a)));
  assert(close(t.paramPoint(a.x,a.y),f(b)));

  // The parametric coordinates of another surface, and of a copy.
  surface r=surface(f,(0,0),(1,1),nu,nv);
  r.domain(s);
  assert(r.paramCoords(1,1.5) == s.paramCoords(1,1.5));
  // With a grid of another size, the domain is the same but not the spacing.
  surface r=surface(f,(0,0),(1,1),2nu,nv+2);
  r.domain(s);
  assert(abs(r.paramCoords(0,0)-a) <= 1e-12);
  assert(abs(r.paramCoords(2nu,nv+2)-b) <= 1e-12);
  assert(abs(r.paramCoords(nu,0)-(3,2)) <= 1e-12);
  assert(close(r.paramPoint(b.x,b.y),f((1,1))));
  // An unstructured surface has no domain to give.
  r.domain(unitsphere);
  assert(abs(r.paramCoords(2nu,nv+2)-b) <= 1e-12);
  r.domain((-1,-1),(0,0));
  assert(s.paramCoords(0,0) == a);
  assert(surface(s).paramCoords(1,1.5) == s.paramCoords(1,1.5));
  assert((shift(Z)*s).paramCoords(1,1.5) == s.paramCoords(1,1.5));

  // A box without area and a surface without patches are left alone.
  t.domain(a,b);
  t.domain((1,2),(1,8));
  t.domain((1,2),(5,2));
  assert(t.paramCoords(0,0) == a);
  surface e;
  e.domain(a,b);
  assert(surface(f,(0,0),(0,1),nu,nv).s.length == nu*nv);
  assert(surface(O,(1,0,0)--(1,0,1),Z,4,30,30).s.length == 4);
}
EndTest();

StartTest("surface domain: logarithmic axes");
{
  // The parametric coordinates of a graph are the x and y coordinates of its
  // points, which on a logarithmic axis are the logarithms of the values
  // plotted.
  picture pic;
  scale(pic,Log,Linear,Log);
  pair a=(1,2), b=(100,8);
  int nx=4, ny=3;
  real f(pair z) {return z.x^2*z.y;}
  real[][] F=new real[nx+1][ny+1];
  for(int i=0; i <= nx; ++i)
    for(int j=0; j <= ny; ++j)
      F[i][j]=f((10^(i/2),2+2j));

  surface[] S={surface(pic,f,a,b,nx,ny),surface(pic,f,a,b,nx,ny,Spline),
               surface(pic,F,a,b),surface(pic,F,a,b,Spline)};
  for(surface s : S) {
    assert(abs(s.paramCoords(0,0)-(0,2)) <= 1e-12);
    assert(abs(s.paramCoords(nx,ny)-(2,8)) <= 1e-12);
    for(int i=0; i <= nx; ++i) {
      for(int j=0; j <= ny; ++j) {
        triple p=s.point(i,j);
        assert(abs(s.paramCoords(i,j)-(p.x,p.y)) <= 1e-12);
        assert(close(p,(i/2,2+2j,log10(f((10^(i/2),2+2j))))));
        assert(close(s.paramPoint(p.x,p.y),p));
      }
    }
    // Between the samples as well, u and v are the x and y coordinates.
    triple p=s.paramPoint(0.8,3.5);
    assert(abs((p.x,p.y)-(0.8,3.5)) <= 1e-9);
  }

  // The parameters of a parametric surface are not coordinates and are not
  // scaled.
  triple g(pair z) {return (10^z.x,z.y,1);}
  for(surface s : new surface[] {surface(pic,g,a,b,nx,ny),
        surface(pic,g,a,b,nx,ny,Spline)}) {
    assert(abs(s.paramCoords(0,0)-a) <= 1e-12);
    assert(abs(s.paramCoords(nx,ny)-b) <= 1e-12);
  }
}
EndTest();

