import TestLib;
import graph3;

bool close(triple a, triple b)
{
  real norm=(b == O) ? 1 : max(abs(a),abs(b));
  return abs(a-b) <= 100*realEpsilon*norm;
}

// A cell (i,j) of a parametric surface grid is "missing" when one of its four
// corner samples is excluded by the construction condition.  We drop the single
// corner vertex (2,2), which removes the four cells touching it.
bool missing(int i, int j)
{
  return (i == 1 || i == 2) && (j == 1 || j == 2);
}

StartTest("missing patch (triple[][] grid)");
{
  int nx=3, ny=3;
  triple[][] f=new triple[nx+1][ny+1];
  for(int i=0; i <= nx; ++i)
    for(int j=0; j <= ny; ++j)
      f[i][j]=(i,j,0);

  bool[][] cond=array(nx+1,array(ny+1,true));
  cond[2][2]=false;

  surface s=surface(f,cond);

  assert(s.index.length == nx);
  assert(s.index[0].length == ny);

  for(int i=0; i < nx; ++i) {
    for(int j=0; j < ny; ++j) {
      if(missing(i,j))
        assert(!s.index[i].initialized(j));
      else
        assert(s.index[i].initialized(j) && s.index[i][j] >= 0);
    }
  }

  // point() on an active cell interior.
  assert(close(s.point(0.5,0.5),(0.5,0.5,0)));

  // point() falls back into a missing cell across a shared lower edge: cell
  // (1,1) is missing but its lower neighbor (0,1) is present.
  assert(close(s.point(1.0,1.5),(1.0,1.5,0)));

  // normal() on an active cell is the unit z direction.
  triple n=s.normal(0.5,0.5);
  assert(close((n.x,n.y),(0,0)));
  assert(close(abs(n),1));
}
EndTest();

StartTest("missing patch (spline parametric)");
{
  triple f(pair z) {return (z.x,z.y,0);}
  bool cond(pair z) {return abs(z.x-2) > 0.5 || abs(z.y-2) > 0.5;}

  surface s=surface(f,(0,0),(3,3),3,3,Spline,Spline,cond);

  assert(s.index.length == 3);
  assert(s.index[0].length == 3);

  for(int i=0; i < 3; ++i) {
    for(int j=0; j < 3; ++j) {
      if(missing(i,j))
        assert(!s.index[i].initialized(j));
      else
        assert(s.index[i].initialized(j) && s.index[i][j] >= 0);
    }
  }

  assert(close(s.point(0.5,0.5),(0.5,0.5,0)));
  assert(close(s.point(1.0,1.5),(1.0,1.5,0)));
}
EndTest();
