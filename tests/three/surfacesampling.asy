import TestLib;
import graph3;

bool close(triple a, triple b)
{
  real norm=max(abs(a),abs(b),1);
  return abs(a-b) <= 1e-9*norm;
}

// The corner of s at the mesh node (i,j), where i < nu and j < nv.
triple node(surface s, int i, int j)
{
  return s.s[s.index[i][j]].point(0,0);
}

// An ellipsoid with x in [10,1000], so that it can be drawn on a logarithmic
// x axis. The parameters are angles, and the first one vanishes at a pole.
triple F(pair t)
{
  return (505+495*sin(t.x)*cos(t.y),2+sin(t.x)*sin(t.y),2+cos(t.x));
}

real g(pair z) {return 1+z.x*z.y^2;}

int nu=6, nv=4;

StartTest("surface sampling: linear constructor from arrays");
{
  real[] u={0,0.1,0.5,2,3};
  real[] v={1,1.5,4};
  surface s=surface(F,u,v);
  assert(s.s.length == (u.length-1)*(v.length-1));
  assert(s.index.length == u.length-1 && s.index[0].length == v.length-1);
  for(int i=0; i < u.length-1; ++i)
    for(int j=0; j < v.length-1; ++j) {
      patch p=s.s[s.index[i][j]];
      assert(p.point(0,0) == F((u[i],v[j])));
      assert(p.point(1,0) == F((u[i+1],v[j])));
      assert(p.point(1,1) == F((u[i+1],v[j+1])));
      assert(p.point(0,1) == F((u[i],v[j+1])));
    }

  // Only cells with four active corners are kept, and f is not evaluated
  // where cond is false.
  bool cond(pair t) {return t.x < 2.5;}
  triple G(pair t) {assert(cond(t)); return F(t);}
  surface s=surface(G,u,v,cond);
  assert(s.s.length == (u.length-2)*(v.length-1));
  assert(node(s,2,1) == F((u[2],v[1])));

  // Too few values for a single cell.
  assert(surface(F,new real[] {1},v).s.length == 0);
  assert(surface(F,u,new real[]).s.length == 0);

  // The constructor over box(a,b) is the same surface for evenly spaced arrays.
  pair a=(0.3,0.7), b=(2.9,3.3);
  surface s=surface(F,a,b,nu,nv);
  surface t=surface(F,uniform(a.x,b.x,nu),uniform(a.y,b.y,nv));
  assert(s.s.length == nu*nv && t.s.length == nu*nv);
  for(int i=0; i < nu; ++i)
    for(int j=0; j < nv; ++j)
      assert(close(node(s,i,j),node(t,i,j)));
}
EndTest();

StartTest("surface sampling: parameters ignore the picture scaling");
{
  picture pic;
  scale(pic,Log,Linear,Log);

  // The parameters are sampled at evenly spaced values, although the x axis
  // is logarithmic; only the points of the surface are scaled. A parameter may
  // be zero or negative.
  pair a=(0,-1), b=(3,5);
  surface linear=surface(pic,F,a,b,nu,nv);
  surface spline=surface(pic,F,a,b,nu,nv,Spline);
  assert(linear.s.length == nu*nv && spline.s.length == nu*nv);
  for(int i=0; i < nu; ++i)
    for(int j=0; j < nv; ++j) {
      pair t=(interp(a.x,b.x,i/nu),interp(a.y,b.y,j/nv));
      triple P=F(t);
      triple scaled=(log10(P.x),P.y,log10(P.z));
      assert(close(node(linear,i,j),scaled));
      assert(close(node(spline,i,j),scaled));
    }
}
EndTest();

StartTest("surface sampling: graphs follow the picture scaling");
{
  picture pic;
  scale(pic,Log,Linear,Log);

  // For the graph of a real function the arguments are coordinates of the
  // picture, so the samples are evenly spaced along its axes.
  pair a=(1,2), b=(1e6,6);
  surface linear=surface(pic,g,a,b,nu,nv);
  surface spline=surface(pic,g,a,b,nu,nv,Spline);
  assert(linear.s.length == nu*nv && spline.s.length == nu*nv);
  for(int i=0; i < nu; ++i)
    for(int j=0; j < nv; ++j) {
      real x=10^(6*i/nu);
      real y=interp(a.y,b.y,j/nv);
      triple scaled=(6*i/nu,y,log10(g((x,y))));
      assert(close(node(linear,i,j),scaled));
      assert(close(node(spline,i,j),scaled));
    }
}
EndTest();
