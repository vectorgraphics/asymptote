import TestLib;
import graph3;

bool close(triple a, triple b)
{
  real norm=max(abs(a),abs(b),1);
  return abs(a-b) <= 1e-6*norm;
}

// True when a and b point along the same line (parallel or antiparallel),
// so the comparison is insensitive to patch orientation.
bool parallel(triple a, triple b)
{
  return abs(abs(dot(unit(a),unit(b)))-1) <= 1e-6;
}

// surface() with a parametric f over box(a,b) sets a non-identity
// paramToSurface; paramNormal() should remap (u,v) through it and then
// delegate to normal().
StartTest("paramNormal on a tilted plane");
{
  // Plane z = 2x - 3y + 1; its unit normal is the constant unit(-2,3,1).
  triple f(pair z) {return (z.x,z.y,2*z.x-3*z.y+1);}

  // Parametrize over a box whose corners differ from the surface grid
  // coordinates, so paramToSurface is a nontrivial scale and shift.
  pair a=(1,2), b=(5,8);
  int nu=4, nv=3;
  surface s=surface(f,a,b,nu,nv);

  assert(s.paramToSurface != identity);

  triple expected=unit((-2,3,1));

  for(int i=0; i <= nu; ++i) {
    for(int j=0; j <= nv; ++j) {
      pair p=(interp(a.x,b.x,i/nu),interp(a.y,b.y,j/nv));

      // Definitional contract: paramNormal remaps through paramToSurface
      // and then calls normal().
      pair sc=s.paramToSurface*p;
      assert(close(s.paramNormal(p.x,p.y),s.normal(sc.x,sc.y)));

      // The plane's unit normal is constant and known.
      assert(parallel(s.paramNormal(p.x,p.y),expected));
    }
  }
}
EndTest();

// On a curved surface the normal varies across the domain, so this test
// fails if paramNormal were to ignore the paramToSurface mapping.
StartTest("paramNormal on a curved surface");
{
  // Parabolic cylinder z = x^2; analytic unit normal is unit(-2x,0,1).
  triple f(pair z) {return (z.x,z.y,z.x^2);}

  pair a=(-1,0), b=(2,3);
  int nu=6, nv=4;
  surface s=surface(f,a,b,nu,nv,Spline);

  assert(s.paramToSurface != identity);

  triple[] normals;
  for(int i=0; i <= nu; ++i) {
    real x=interp(a.x,b.x,i/nu);
    pair p=(x,(a.y+b.y)/2);

    // Definitional contract, exact regardless of interpolation error.
    pair sc=s.paramToSurface*p;
    assert(close(s.paramNormal(p.x,p.y),s.normal(sc.x,sc.y)));

    // Spline normal closely matches the analytic normal at grid nodes.
    assert(parallel(s.paramNormal(p.x,p.y),(-2*x,0,1)));

    normals.push(s.paramNormal(p.x,p.y));
  }

  // The normal genuinely varies, so the mapping is being exercised.
  assert(!close(normals[0],normals[normals.length-1]));
}
EndTest();
