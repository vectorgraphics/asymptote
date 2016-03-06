// Asymptote mathematics routines

int quadrant(real degrees)
{
  return floor(degrees/90) % 4;
}

// Roots of unity.
pair unityroot(int n, int k=1)
{
  return expi(2pi*k/n);
}

real csc(real x) {return 1/sin(x);}
real sec(real x) {return 1/cos(x);}
real cot(real x) {return tan(pi/2-x);}

real acsc(real x) {return asin(1/x);}
real asec(real x) {return acos(1/x);}
real acot(real x) {return pi/2-atan(x);}

real frac(real x) {return x-(int)x;}

pair exp(explicit pair z) {return exp(z.x)*expi(z.y);}
pair log(explicit pair z) {return log(abs(z))+I*angle(z);}

// Return an Nx by Ny unit square lattice with lower-left corner at (0,0).
picture grid(int Nx, int Ny, pen p=currentpen)
{
  picture pic;
  for(int i=0; i <= Nx; ++i) draw(pic,(i,0)--(i,Ny),p);
  for(int j=0; j <= Ny; ++j) draw(pic,(0,j)--(Nx,j),p);
  return pic; 
}

bool polygon(path p)
{
  return cyclic(p) && piecewisestraight(p);
}

// Return the intersection time of the point on the line through p and q
// that is closest to z.
real intersect(pair p, pair q, pair z)
{
  pair u=q-p;
  real denom=dot(u,u);
  return denom == 0 ? infinity : dot(z-p,u)/denom;
}

// Return the intersection time of the extension of the line segment PQ
// with the plane perpendicular to n and passing through Z.
real intersect(triple P, triple Q, triple n, triple Z)
{
  real d=n.x*Z.x+n.y*Z.y+n.z*Z.z;
  real denom=n.x*(Q.x-P.x)+n.y*(Q.y-P.y)+n.z*(Q.z-P.z);
  return denom == 0 ? infinity : (d-n.x*P.x-n.y*P.y-n.z*P.z)/denom;
}
                    
// Return any point on the intersection of the two planes with normals
// n0 and n1 passing through points P0 and P1, respectively.
// If the planes are parallel return (infinity,infinity,infinity).
triple intersectionpoint(triple n0, triple P0, triple n1, triple P1)
{
  real Dx=n0.y*n1.z-n1.y*n0.z;
  real Dy=n0.z*n1.x-n1.z*n0.x;
  real Dz=n0.x*n1.y-n1.x*n0.y;
  if(abs(Dx) > abs(Dy) && abs(Dx) > abs(Dz)) {
    Dx=1/Dx;
    real d0=n0.y*P0.y+n0.z*P0.z;
    real d1=n1.y*P1.y+n1.z*P1.z+n1.x*(P1.x-P0.x);
    real y=(d0*n1.z-d1*n0.z)*Dx;
    real z=(d1*n0.y-d0*n1.y)*Dx;
    return (P0.x,y,z);
  } else if(abs(Dy) > abs(Dz)) {
    Dy=1/Dy;
    real d0=n0.z*P0.z+n0.x*P0.x;
    real d1=n1.z*P1.z+n1.x*P1.x+n1.y*(P1.y-P0.y);
    real z=(d0*n1.x-d1*n0.x)*Dy;
    real x=(d1*n0.z-d0*n1.z)*Dy;
    return (x,P0.y,z);
  } else {
    if(Dz == 0) return (infinity,infinity,infinity);
    Dz=1/Dz;
    real d0=n0.x*P0.x+n0.y*P0.y;
    real d1=n1.x*P1.x+n1.y*P1.y+n1.z*(P1.z-P0.z);
    real x=(d0*n1.y-d1*n0.y)*Dz;
    real y=(d1*n0.x-d0*n1.x)*Dz;
    return (x,y,P0.z);
  }
}

// Given a real array a, return its partial sums.
real[] partialsum(real[] a)
{
  real[] b=new real[a.length];
  real sum=0;
  for(int i=0; i < a.length; ++i) {
    sum += a[i];
    b[i]=sum;
  }
  return b;
}

// Given a real array a, return its partial dx-weighted sums.
real[] partialsum(real[] a, real[] dx)
{
  real[] b=new real[a.length];
  real sum=0;
  for(int i=0; i < a.length; ++i) {
    sum += a[i]*dx[i];
    b[i]=sum;
  }
  return b;
}

// Given an integer array a, return its partial sums.
int[] partialsum(int[] a)
{
  int[] b=new int[a.length];
  int sum=0;
  for(int i=0; i < a.length; ++i) {
    sum += a[i];
    b[i]=sum;
  }
  return b;
}

// Given an integer array a, return its partial dx-weighted sums.
int[] partialsum(int[] a, int[] dx)
{
  int[] b=new int[a.length];
  int sum=0;
  for(int i=0; i < a.length; ++i) {
    sum += a[i]*dx[i];
    b[i]=sum;
  }
  return b;
}

// If strict=false, return whether i > j implies a[i] >= a[j]
// If strict=true, return whether  i > j implies a[i] > a[j]
bool increasing(real[] a, bool strict=false)
{
  real[] ap=copy(a);
  ap.delete(0);
  ap.push(0);
  bool[] b=strict ? (ap > a) : (ap >= a);
  b[a.length-1]=true;
  return all(b);
}

// Return the first and last indices of consecutive true-element segments
// of bool[] b.
int[][] segmentlimits(bool[] b)
{
  int[][] segment;
  bool[] n=copy(b);
  n.delete(0);
  n.push(!b[b.length-1]);
  int[] edge=(b != n) ? sequence(1,b.length) : null;
  edge.insert(0,0);
  int stop=edge[0];
  for(int i=1; i < edge.length; ++i) {
    int start=stop;
    stop=edge[i];
    if(b[start])
      segment.push(new int[] {start,stop-1});
  }
  return segment;
}

// Return the indices of consecutive true-element segments of bool[] b.
int[][] segment(bool[] b)
{
  int[][] S=segmentlimits(b);
  return sequence(new int[](int i) {
      return sequence(S[i][0],S[i][1]);
    },S.length);
}

// If the sorted array a does not contain x, insert it sequentially,
// returning the index of x in the resulting array.
int unique(real[] a, real x) {
  int i=search(a,x);
  if(i == -1 || x != a[i]) {
    ++i;
    a.insert(i,x);
  }
  return i;
}

int unique(string[] a, string x) {
  int i=search(a,x);
  if(i == -1 || x != a[i]) {
    ++i;
    a.insert(i,x);
  }
  return i;
}

bool lexorder(pair a, pair b) {
  return a.x < b.x || (a.x == b.x && a.y < b.y);
}

bool lexorder(triple a, triple b) {
  return a.x < b.x || (a.x == b.x && (a.y < b.y || (a.y == b.y && a.z < b.z)));
}

real[] zero(int n)
{
  return sequence(new real(int) {return 0;},n);
}

real[][] zero(int n, int m)
{
  real[][] M=new real[n][];
  for(int i=0; i < n; ++i)
    M[i]=sequence(new real(int) {return 0;},m);
  return M;
}

bool square(real[][] m)
{
  int n=m.length;
  for(int i=0; i < n; ++i)
    if(m[i].length != n) return false;
  return true;
}

bool rectangular(real[][] m)
{
  int n=m.length;
  if(n > 0) {
    int m0=m[0].length;
    for(int i=1; i < n; ++i)
      if(m[i].length != m0) return false;
  }
  return true;
}

bool rectangular(pair[][] m)
{
  int n=m.length;
  if(n > 0) {
    int m0=m[0].length;
    for(int i=1; i < n; ++i)
      if(m[i].length != m0) return false;
  }
  return true;
}

bool rectangular(triple[][] m)
{
  int n=m.length;
  if(n > 0) {
    int m0=m[0].length;
    for(int i=1; i < n; ++i)
      if(m[i].length != m0) return false;
  }
  return true;
}

// draw the (infinite) line going through P and Q, without altering the
// size of picture pic.
void drawline(picture pic=currentpicture, pair P, pair Q, pen p=currentpen)
{
  pic.add(new void (frame f, transform t, transform T, pair m, pair M) {
      // Reduce the bounds by the size of the pen.
      m -= min(p); M -= max(p);

      // Calculate the points and direction vector in the transformed space.
      t=t*T;
      pair z=t*P;
      pair v=t*Q-z;

      // Handle horizontal and vertical lines.
      if(v.x == 0) {
        if(m.x <= z.x && z.x <= M.x)
          draw(f,(z.x,m.y)--(z.x,M.y),p);
      } else if(v.y == 0) {
        if(m.y <= z.y && z.y <= M.y)
          draw(f,(m.x,z.y)--(M.x,z.y),p);
      } else {
        // Calculate the maximum and minimum t values allowed for the
        // parametric equation z + t*v
        real mx=(m.x-z.x)/v.x, Mx=(M.x-z.x)/v.x;
        real my=(m.y-z.y)/v.y, My=(M.y-z.y)/v.y;
        real tmin=max(v.x > 0 ? mx : Mx, v.y > 0 ? my : My);
        real tmax=min(v.x > 0 ? Mx : mx, v.y > 0 ? My : my);
        if(tmin <= tmax)
          draw(f,z+tmin*v--z+tmax*v,p);
      }
    },true);
}

real interpolate(real[] x, real[] y, real x0, int i) 
{
  int n=x.length;
  if(n == 0) abort("Zero data points in interpolate");
  if(n == 1) return y[0];
  if(i < 0) {
    real dx=x[1]-x[0];
    return y[0]+(y[1]-y[0])/dx*(x0-x[0]);
  }
  if(i >= n-1) {
    real dx=x[n-1]-x[n-2];
    return y[n-1]+(y[n-1]-y[n-2])/dx*(x0-x[n-1]);
  }

  real D=x[i+1]-x[i];
  real B=(x0-x[i])/D;
  real A=1.0-B;
  return A*y[i]+B*y[i+1];
}

// Linearly interpolate data points (x,y) to (x0,y0), where the elements of
// real[] x are listed in ascending order and return y0. Values outside the
// available data range are linearly extrapolated using the first derivative
// at the nearest endpoint.
real interpolate(real[] x, real[] y, real x0) 
{
  return interpolate(x,y,x0,search(x,x0));
}

private string nopoint="point not found";

// Return the nth intersection time of path g with the vertical line through x.
real time(path g, real x, int n=0)
{
  real[] t=times(g,x);
  if(t.length <= n) abort(nopoint);
  return t[n];
}

// Return the nth intersection time of path g with the horizontal line through
// (0,z.y).
real time(path g, explicit pair z, int n=0)
{
  real[] t=times(g,z);
  if(t.length <= n) abort(nopoint);
  return t[n];
}

// Return the nth y value of g at x.
real value(path g, real x, int n=0)
{
  return point(g,time(g,x,n)).y;
}

// Return the nth x value of g at y=z.y.
real value(path g, explicit pair z, int n=0)
{
  return point(g,time(g,(0,z.y),n)).x;
}

// Return the nth slope of g at x.
real slope(path g, real x, int n=0)
{
  pair a=dir(g,time(g,x,n));
  return a.y/a.x;
}

// Return the nth slope of g at y=z.y.
real slope(path g, explicit pair z, int n=0)
{
  pair a=dir(g,time(g,(0,z.y),n));
  return a.y/a.x;
}

// A quartic complex root solver based on these references:
// http://planetmath.org/encyclopedia/GaloisTheoreticDerivationOfTheQuarticFormula.html
// Neumark, S., Solution of Cubic and Quartic Equations, Pergamon Press
// Oxford (1965).
pair[] quarticroots(real a, real b, real c, real d, real e)
{
  real Fuzz=100000*realEpsilon;

  // Remove roots at numerical infinity.
  if(abs(a) <= Fuzz*(abs(b)+Fuzz*(abs(c)+Fuzz*(abs(d)+Fuzz*abs(e)))))
    return cubicroots(b,c,d,e);
  
  // Detect roots at numerical zero.
  if(abs(e) <= Fuzz*(abs(d)+Fuzz*(abs(c)+Fuzz*(abs(b)+Fuzz*abs(a)))))
    return cubicroots(a,b,c,d);

  real ainv=1/a;
  b *= ainv;
  c *= ainv;
  d *= ainv;
  e *= ainv;
  
  pair[] roots;
  real[] T=cubicroots(1,-2c,c^2+b*d-4e,d^2+b^2*e-b*c*d);
  if(T.length == 0) return roots;

  real t0=T[0];
  pair[] sum=quadraticroots((1,0),(b,0),(t0,0));
  pair[] product=quadraticroots((1,0),(t0-c,0),(e,0));

  if(abs(sum[0]*product[0]+sum[1]*product[1]+d) <
     abs(sum[0]*product[1]+sum[1]*product[0]+d))
    product=reverse(product);

  for(int i=0; i < 2; ++i)
    roots.append(quadraticroots((1,0),-sum[i],product[i]));

  return roots;
}

pair[][] fft(pair[][] a, int sign=1)
{
  pair[][] A=new pair[a.length][];
  int k=0;
  for(pair[] v : a) {
    A[k]=fft(v,sign);
    ++k;
  }
  a=transpose(A);
  k=0;
  for(pair[] v : a) {
    A[k]=fft(v,sign);
    ++k;
  }
  return transpose(A);
}

// Given a matrix A with independent columns, return
// the unique vector y minimizing |Ay - b|^2 (the L2 norm).
// If the columns of A are not linearly independent,
// throw an error (if warn == true) or return an empty array
// (if warn == false).
real[] leastsquares(real[][] A, real[] b, bool warn=true)
{
  real[] solution=solve(AtA(A),b*A,warn=false);
  if (solution.length == 0 && warn)
    abort("Cannot compute least-squares approximation for " +
	  "a matrix with linearly dependent columns.");
  return solution;
}

// Namespace
struct rootfinder_settings {
  static real roottolerance = 1e-4;
}

real findroot(real f(real), real a, real b,
              real tolerance=rootfinder_settings.roottolerance,
              real fa=f(a), real fb=f(b))
{
  return _findroot(f,a,b,tolerance,fa,fb);
}
