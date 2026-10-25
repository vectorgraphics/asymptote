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

// surface() with a parametric f over box(a,b) has parametric coordinates
// that differ from its surface coordinates; paramNormal() should map (u,v)
// from the former to the latter and then delegate to normal().
StartTest("paramNormal on a tilted plane");
{
  // Plane z = 2x - 3y + 1; its unit normal is the constant unit(-2,3,1).
  triple f(pair z) {return (z.x,z.y,2*z.x-3*z.y+1);}

  // Parametrize over a box whose corners differ from the surface grid
  // coordinates, so the map between them is a nontrivial scale and shift.
  pair a=(1,2), b=(5,8);
  int nu=4, nv=3;
  surface s=surface(f,a,b,nu,nv);

  assert(s.paramCoords(0,0) == a);

  triple expected=unit((-2,3,1));

  for(int i=0; i <= nu; ++i) {
    for(int j=0; j <= nv; ++j) {
      pair p=(interp(a.x,b.x,i/nu),interp(a.y,b.y,j/nv));

      // Definitional contract: paramNormal at the parametric coordinates
      // of the mesh node (i,j) is normal() there.
      assert(close(s.paramCoords(i,j),p));
      assert(close(s.paramNormal(p.x,p.y),s.normal(i,j)));

      // The plane's unit normal is constant and known.
      assert(parallel(s.paramNormal(p.x,p.y),expected));
    }
  }
}
EndTest();

// On a curved surface the normal varies across the domain, so this test
// fails if paramNormal were to ignore the parametric coordinates.
StartTest("paramNormal on a curved surface");
{
  // Parabolic cylinder z = x^2; analytic unit normal is unit(-2x,0,1).
  triple f(pair z) {return (z.x,z.y,z.x^2);}

  pair a=(-1,0), b=(2,3);
  int nu=6, nv=4;
  surface s=surface(f,a,b,nu,nv,Spline);

  assert(s.paramCoords(0,0) == a);

  triple[] normals;
  for(int i=0; i <= nu; ++i) {
    real x=interp(a.x,b.x,i/nu);
    pair p=(x,(a.y+b.y)/2);

    // Definitional contract, exact regardless of interpolation error.
    assert(close(s.paramNormal(p.x,p.y),s.normal(i,nv/2)));

    // Spline normal closely matches the analytic normal at grid nodes.
    assert(parallel(s.paramNormal(p.x,p.y),(-2*x,0,1)));

    normals.push(s.paramNormal(p.x,p.y));
  }

  // The normal genuinely varies, so the mapping is being exercised.
  assert(!close(normals[0],normals[normals.length-1]));
}
EndTest();
