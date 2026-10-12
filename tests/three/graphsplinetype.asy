import TestLib;
import graph3;

// The largest distance between corresponding control points of a and b.
real distance(surface a, surface b)
{
  assert(a.s.length == b.s.length);
  real m=0;
  for(int k=0; k < a.s.length; ++k)
    for(int i=0; i < 4; ++i)
      for(int j=0; j < 4; ++j)
        m=max(m,abs(a.s[k].P[i][j]-b.s[k].P[i][j]));
  return m;
}

bool same(surface a, surface b) {return distance(a,b) < 1e-12;}

int nx=8, ny=6;

StartTest("graph splinetype: Spline is notaknot");
{
  // Periodic end conditions are never chosen from the data, even for a
  // function that is periodic in both directions.
  real f(pair z) {return cos(z.x)*cos(z.y);}
  pair a=(0,0), b=(2pi,2pi);
  surface s=surface(f,a,b,nx,nx,Spline);
  assert(same(s,surface(f,a,b,nx,nx,notaknot)));
  // They can be requested.
  assert(!same(s,surface(f,a,b,nx,nx,periodic)));
  assert(!same(s,surface(f,a,b,nx,nx,periodic,notaknot)));
}
EndTest();

StartTest("graph splinetype: matrix with arrays");
{
  // Unevenly spaced x, with one period of a function of x.
  real[] x={0,0.1,0.25,0.5,0.6,0.8,0.9,1};
  real[] y={0,1,3};
  real[][] f=new real[x.length][y.length];
  for(int i=0; i < x.length; ++i)
    for(int j=0; j < y.length; ++j)
      f[i][j]=sin(2pi*x[i])+y[j];
  // The default is Spline.
  surface s=surface(f,x,y);
  assert(same(s,surface(f,x,y,Spline)));
  assert(same(s,surface(f,x,y,notaknot)));
  assert(!same(s,surface(f,x,y,periodic,notaknot)));

  // The choice does not depend on the relative size of x and y.
  real[] Y=1e9*y;
  assert(same(surface(f,x,Y),surface(f,x,Y,notaknot)));
  real[][] g=copy(f);
  g[x.length-1]=f[0]+1;
  assert(same(surface(g,x,Y),surface(g,x,Y,notaknot)));
}
EndTest();

StartTest("graph splinetype: a graph is not cyclic");
{
  real f(pair z) {return cos(z.x)*cos(z.y);}
  pair a=(0,0), b=(2pi,2pi);
  for(surface s : new surface[] {surface(f,a,b,nx,nx,Spline),
                                 surface(f,a,b,nx,nx,periodic)}) {
    assert(!s.ucyclic() && !s.vcyclic());
    path3 g=s.vequals(1);
    assert(!cyclic(g) && length(g) == nx);
    path3 g=s.uequals(1);
    assert(!cyclic(g) && length(g) == nx);
  }
}
EndTest();
