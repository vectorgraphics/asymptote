import TestLib;
import graph3;
import palette;

bool close(triple a, triple b)
{
  real norm=max(abs(a),abs(b),1);
  return abs(a-b) <= 1e-9*norm;
}

bool close(pair a, pair b)
{
  real norm=max(abs(a),abs(b),1);
  return abs(a-b) <= 1e-9*norm;
}

transform3 pictureTransform=shift(1,1,1)*scale3(2);

// Draw s into a scratch picture and return the surfaceVertex passed to the
// vertexPen for each patch corner, in the order of the calls. Nothing is
// rendered or written to disk. The picture is fit with a transform other
// than the identity: the pen must still see the coordinates of s itself.
surfaceVertex[] collect(surface s, render render=defaultrender)
{
  surfaceVertex[] v;
  picture pic;
  draw(pic,s,new pen(surfaceVertex sv) {v.push(sv); return red;},
       render=render);
  pic.fit3(pictureTransform,null,currentprojection);
  return v;
}

// The corners of a patch in the order used by surfaceVertex.corner.
triple[] corners(patch p)
{
  triple[] c={p.P[0][0],p.P[3][0],p.P[3][3]};
  if(!p.triangular) c.push(p.P[0][3]);
  return c;
}

pair[] cornerOffset={(0,0),(1,0),(1,1),(0,1)};

// Check everything that holds for every structured surface: each patch that
// exists is visited once, corner by corner, with its own index and cell, and
// uv is the parametric coordinate at which the surface passes through z.
surfaceVertex[] checkStructured(surface s)
{
  surfaceVertex[] v=collect(s);
  assert(v.length == 4*s.s.length);
  bool[] seen=array(s.s.length,false);
  for(int k=0; k < v.length; ++k) {
    surfaceVertex sv=v[k];
    assert(sv.corner == k % 4);
    assert(sv.U >= 0 && sv.U < s.index.length);
    assert(sv.V >= 0 && sv.V < s.index[0].length);
    assert(s.index[sv.U].initialized(sv.V));
    assert(s.index[sv.U][sv.V] == sv.patch);
    if(sv.corner == 0) {
      assert(!seen[sv.patch]);
      seen[sv.patch]=true;
    } else {
      // All corners of a patch report the same patch and cell.
      assert(sv.patch == v[k-1].patch);
      assert(sv.U == v[k-1].U && sv.V == v[k-1].V);
    }
    assert(sv.z == corners(s.s[sv.patch])[sv.corner]);
    pair node=(sv.U,sv.V)+cornerOffset[sv.corner];
    assert(sv.uv == s.paramCoords(node.x,node.y));
    assert(close(s.paramPoint(sv.uv.x,sv.uv.y),sv.z));
  }
  assert(all(seen));
  // The pen colors a copy made at draw time, never the surface itself.
  for(patch p : s.s)
    assert(p.colors.length == 0);
  return v;
}

StartTest("vertexPen: parametric surface");
{
  triple f(pair z) {return (z.x+z.y,z.x-z.y,z.x*z.y);}
  pair a=(1,2), b=(5,8);
  int nu=2, nv=3;

  surfaceVertex[] v=checkStructured(surface(f,a,b,nu,nv));
  // The exact values for the first patch, including corner 0.
  pair[] uv={(1,2),(3,2),(3,4),(1,4)};
  for(int j=0; j < 4; ++j) {
    assert(v[j].patch == 0 && v[j].corner == j && v[j].U == 0 && v[j].V == 0);
    assert(close(v[j].uv,uv[j]));
    assert(close(v[j].z,f(uv[j])));
  }
  for(surfaceVertex sv : v) {
    assert(sv.patch == nv*sv.U+sv.V);
    assert(close(sv.z,f(sv.uv)));
  }

  for(surfaceVertex sv : checkStructured(surface(f,a,b,nu,nv,Spline)))
    assert(close(sv.z,f(sv.uv)));

  // With a > b the parametrization is reflected, not rejected.
  for(surfaceVertex sv : checkStructured(surface(f,b,a,nu,nv)))
    assert(close(sv.z,f(sv.uv)));
}
EndTest();

StartTest("vertexPen: graph of a real function");
{
  real f(pair z) {return z.x^2-z.y;}
  pair a=(-1,0), b=(2,3);

  for(surfaceVertex sv : checkStructured(surface(f,a,b,3,2)))
    assert(close(sv.z,(sv.uv.x,sv.uv.y,f(sv.uv))));

  // The spline form is parametrized over box(a,b) too, not over the mesh.
  surfaceVertex[] v=checkStructured(surface(f,a,b,3,2,Spline));
  for(surfaceVertex sv : v)
    assert(close(sv.z,(sv.uv.x,sv.uv.y,f(sv.uv))));
  assert(close(v[0].uv,a));
  assert(close(v[v.length-2].uv,b));
}
EndTest();

StartTest("vertexPen: matrix surfaces");
{
  // A matrix of heights over box(a,b) is parametrized over box(a,b), with
  // or without a splinetype.
  real[][] h={{0,1,2},{1,2,3},{2,3,5},{4,4,4}};
  pair a=(10,20), b=(40,60);
  for(surface s : new surface[] {surface(h,a,b),surface(h,a,b,Spline),
        surface(h,a,b,linear)}) {
    for(surfaceVertex sv : checkStructured(s)) {
      pair ij=(sv.U,sv.V)+cornerOffset[sv.corner];
      assert(close(sv.uv,(10+10*ij.x,20+20*ij.y)));
      assert(close(sv.z,(sv.uv.x,sv.uv.y,h[round(ij.x)][round(ij.y)])));
    }
    assert(close(s.paramPoint(40,60),(40,60,4)));
  }
  // Also when a exceeds b.
  for(surfaceVertex sv : checkStructured(surface(h,(40,20),(10,60))))
    assert(close(sv.z.x,sv.uv.x) && close(sv.z.y,sv.uv.y));

  // A matrix sampled at given x and y values, which need not be uniform, is
  // parametrized by mesh coordinates.
  real[] x={10,20,30,40}, y={20,30,60};
  for(surfaceVertex sv : checkStructured(surface(h,x,y))) {
    pair ij=(sv.U,sv.V)+cornerOffset[sv.corner];
    assert(close(sv.uv,ij));
    assert(close(sv.z,(x[round(ij.x)],y[round(ij.y)],
                       h[round(ij.x)][round(ij.y)])));
  }

  // Omitted cells are skipped and the patches that remain keep their own
  // index and cell.
  int nx=3, ny=3;
  triple[][] f=new triple[nx+1][ny+1];
  for(int i=0; i <= nx; ++i)
    for(int j=0; j <= ny; ++j)
      f[i][j]=(i,j,i*j);
  bool[][] cond=array(nx+1,array(ny+1,true));
  cond[2][2]=false;
  surface s=surface(f,cond);
  surfaceVertex[] v=checkStructured(s);
  assert(v.length == 4*5);
  for(surfaceVertex sv : v) {
    assert(!((sv.U == 1 || sv.U == 2) && (sv.V == 1 || sv.V == 2)));
    assert(close(sv.z,(sv.uv.x,sv.uv.y,sv.uv.x*sv.uv.y)));
  }
}
EndTest();

StartTest("vertexPen: surface of revolution");
{
  path3 g=(1,0,0)--(1,0,1)--(2,0,2);
  int n=4;
  surface s=surface(O,g,Z,n,30,120);
  surfaceVertex[] v=checkStructured(s);
  assert(v.length == 4*n*length(g));
  real umin=infinity, umax=-infinity;
  for(surfaceVertex sv : v) {
    // u is the angle of rotation in radians; v is the node of g.
    assert(close(sv.z,rotate(degrees(sv.uv.x),Z)*point(g,sv.uv.y)));
    umin=min(umin,sv.uv.x);
    umax=max(umax,sv.uv.x);
  }
  assert(close(umin,radians(30)));
  assert(close(umax,radians(120)));

  // Both limits of the domain occur in a cyclic direction.
  surface s=surface(O,g,Z,n);
  assert(s.index.cyclic);
  real umin=infinity, umax=-infinity;
  for(surfaceVertex sv : checkStructured(s)) {
    umin=min(umin,sv.uv.x);
    umax=max(umax,sv.uv.x);
  }
  assert(close(umin,0));
  assert(close(umax,2pi));
}
EndTest();

StartTest("vertexPen: unstructured surfaces");
{
  void check(surface s) {
    assert(s.index.length == 0);
    surfaceVertex[] v=collect(s);
    int k=0;
    for(int i=0; i < s.s.length; ++i) {
      triple[] c=corners(s.s[i]);
      for(int j=0; j < c.length; ++j) {
        surfaceVertex sv=v[k];
        assert(sv.z == c[j]);
        assert(sv.patch == i && sv.corner == j);
        assert(sv.uv == (0,0) && sv.U == 0 && sv.V == 0);
        ++k;
      }
    }
    assert(k == v.length);
  }
  check(unitsphere);
  check(surface(surface(unitsquare3),surface(shift(Z)*unitsquare3)));

  // A Bezier triangle has three corners.
  surface t=surface(X--Y--Z--cycle);
  assert(t.s[0].triangular);
  assert(collect(t).length == 3);
  check(t);
}
EndTest();

StartTest("vertexPen: transformed and tessellated surfaces");
{
  triple f(pair z) {return (z.x,z.y,z.x*z.y);}
  pair a=(1,2), b=(5,8);
  surface s=surface(f,a,b,2,2);

  // The pen sees the coordinates of the surface that was drawn, with its
  // parametrization unchanged.
  transform3 T=shift(1,2,3)*rotate(90,Z);
  for(surfaceVertex sv : checkStructured(T*s))
    assert(close(sv.z,T*f(sv.uv)));

  // A copy keeps its parametrization.
  for(surfaceVertex sv : checkStructured(surface(s)))
    assert(close(sv.z,f(sv.uv)));

  // The tessellated path passes the pen the same values.
  surfaceVertex[] v=collect(s);
  surfaceVertex[] w=collect(s,render(tessellate=true));
  assert(w.length == v.length);
  for(int k=0; k < v.length; ++k) {
    assert(w[k].z == v[k].z && close(w[k].uv,v[k].uv));
    assert(w[k].patch == v[k].patch && w[k].corner == v[k].corner);
    assert(w[k].U == v[k].U && w[k].V == v[k].V);
  }
  for(patch p : s.s)
    assert(p.colors.length == 0);
}
EndTest();

StartTest("vertexPen: cast from pen(triple)");
{
  pen byHeight(triple v) {return v.z > 0 ? red : blue;}
  vertexPen vp=byHeight;
  assert(vp(surfaceVertex((0,0,1),(7,8),3,2,1,1)) == red);
  assert(vp(surfaceVertex((0,0,-1))) == blue);

  // A pen(triple) is accepted directly by draw.
  triple[] z;
  picture pic;
  draw(pic,surface(unitsquare3),new pen(triple v) {z.push(v); return red;});
  pic.fit3(pictureTransform,null,currentprojection);
  assert(z.length == 4);
  assert(z[0] == (0,0,0) && z[1] == (1,0,0) && z[2] == (1,1,0) &&
         z[3] == (0,1,0));
}
EndTest();

StartTest("cornerPen");
{
  surfaceVertex at(int patch, int corner) {
    return surfaceVertex(O,patch=patch,corner=corner);
  }

  // The rest form applies one row of pens to every patch.
  pen[] row={red,green,blue,black};
  vertexPen vp=cornerPen(...row);
  for(int i=0; i < 3; ++i)
    for(int j=0; j < 4; ++j)
      assert(vp(at(i,j)) == row[j]);

  // The matrix form takes row i % p.length for patch i.
  pen[][] p={{red,green,blue,black},{cyan,magenta,yellow,gray}};
  vertexPen vp=cornerPen(p);
  for(int i=0; i < 5; ++i)
    for(int j=0; j < 4; ++j)
      assert(vp(at(i,j)) == p[i % 2][j]);

  // A cyclic row is indexed modulo its length; the other rows are not
  // affected.
  pen[] two={red,blue};
  two.cyclic=true;
  pen[][] q={two,{green,green,green,black}};
  vertexPen vp=cornerPen(q);
  assert(vp(at(0,0)) == red && vp(at(0,1)) == blue);
  assert(vp(at(0,2)) == red && vp(at(0,3)) == blue);
  assert(vp(at(1,3)) == black);
  assert(vp(at(2,3)) == blue);

  // The caller's array is not mutated...
  assert(!q.cyclic && q.length == 2);
  assert(q[0].cyclic && !q[1].cyclic);
  assert(q[0].length == 2 && q[0][0] == red && q[0][1] == blue);
  // ...nor aliased, at either level.
  q[0][0]=yellow;
  q[1]=new pen[] {cyan,cyan,cyan,cyan};
  q.push(new pen[] {magenta});
  assert(vp(at(0,0)) == red);
  assert(vp(at(1,3)) == black);
  assert(vp(at(2,3)) == blue);
  row[0]=yellow;
  assert(cornerPen(...row)(at(0,0)) == yellow);

  // Corner j is node j of the path a patch was built from.
  path3 g=(0,0,0)--(2,0,0)--(2,1,1)--(0,1,1)--cycle;
  for(surfaceVertex sv : collect(surface(g)))
    assert(close(sv.z,point(g,sv.corner)));
}
EndTest();

StartTest("palette as a vertexPen");
{
  pen[] pal={red,green,blue,black,white};
  real height(triple v) {return v.z;}
  surfaceVertex at(real z) {return surfaceVertex((0,0,z));}

  vertexPen vp=palette(height,10,18,pal);
  assert(vp(at(10)) == red);
  assert(vp(at(12)) == green);
  assert(vp(at(14.9)) == blue);
  assert(vp(at(18)) == white);

  // A degenerate range selects the first pen; an empty palette gives nullpen.
  assert(palette(height,3,3,pal)(at(3)) == red);
  assert(palette(height,0,1,new pen[])(at(0.5)) == nullpen);

  // The surface method takes the range of f over the patch corners.
  surface s=surface(new triple(pair z) {return (z.x,z.y,z.x+2*z.y);},
                    (0,0),(2,3),2,3);
  assert(close(s.min(height),0));
  assert(close(s.max(height),8));
  vertexPen vp=s.palette(height,pal);
  assert(vp(at(0)) == red);
  assert(vp(at(4)) == blue);
  assert(vp(at(8)) == white);
  for(surfaceVertex sv : collect(s))
    assert(vp(sv) == pal[round(sv.z.z/2)]);

  // The bounds of a surface are available as methods and as functions.
  assert(close(s.min(),(0,0,0)) && close(min(s),(0,0,0)));
  assert(close(s.max(),(2,3,8)) && close(max(s),(2,3,8)));
}
EndTest();
