// Asymptote mathematics routines

public real perpsize=arrowsize;

real abs(pair z)
{
  return length(z);
}

int sgn(real x)
{
  return (x == 0 ? 0 : (x > 0 ? 1 : -1));
}

real radians(real degrees)
{
  return degrees*pi/180;
}

real degrees(real radians) 
{
  return radians*180/pi;
}

int quadrant(real degrees)
{
  return floor(degrees/90) % 4;
}

// Roots of unity. 
pair zeta(int n, int k = 1)
{
  return expi(2pi*k/n);
}

real Sin(real deg) {return sin(radians(deg));}
real Cos(real deg) {return cos(radians(deg));}
real Tan(real deg) {return tan(radians(deg));}
real csc(real x) {return 1/sin(x);}
real sec(real x) {return 1/cos(x);}
real cot(real x) {return tan(pi/2-x);}
real frac(real x) {return x-(int)x;}

// Given a vector A, return its partial (optionally dx-weighted) sums.
real[] partialsum(real[] A, real[] dx=null) 
{
  real[] B=new real[];
  B[0]=0;
  if(alias(dx,null))
    for(int i=0; i < A.length; ++i) B[i+1]=B[i]+A[i];
  else
    for(int i=0; i < A.length; ++i) B[i+1]=B[i]+A[i]*dx[i];
  return B;
}

void perpendicular(picture pic=currentpicture, pair z1, pair z2, real
		   size=perpsize, pen p=currentpen)
{
  pair v=perpsize*unit(z2-z1);
  picture apic=new picture;
  _draw(apic,v--v+I*v--I*v,p);
  addabout(pic,apic,z1);
}

real dotproduct(pair z, pair w) 
{
  return z.x*w.x+z.y*w.y;
}

bool straight(guide p)
{
  for(int i=0; i < length(p); ++i)
    if(!straight(p,i)) return false;
  return true;
}

bool polygon(guide p)
{
  return cyclic(p) && straight(p);
}

// Returns true iff the point z lies in the region bounded by the cyclic
// polygon p.
bool inside(pair z, guide p)
{
  if(!polygon(p)) abort("Polygon must be a straight cyclic path");
  bool c=false;
  int n=length(p);
  for(int i=0; i < n; ++i) {
    pair pi=point(p,i);
    pair pj=point(p,i+1);
    if(((pi.y <= z.y && z.y < pj.y) || (pj.y <= z.y && z.y < pi.y)) &&
       z.x < pi.x+(pj.x-pi.x)*(z.y-pi.y)/(pj.y-pi.y)) c=!c;
  }
  return c;
}

// Returns true iff the line a--b intersects the cyclic polygon p.
bool intersect(pair a, pair b, path p)
{
  if(!polygon(p)) abort("Polygon must be a straight cyclic path");
  int n=length(p);
  for(int i=0; i < n; ++i) {
    pair A=point(p,i);
    pair B=point(p,i+1);
    real de=(b.x-a.x)*(A.y-B.y)-(A.x-B.x)*(b.y-a.y);
    if(de != 0) {
      de=1/de;
      real t=((A.x-a.x)*(A.y-B.y)-(A.x-B.x)*(A.y-a.y))*de;
      real T=((b.x-a.x)*(A.y-a.y)-(A.x-a.x)*(b.y-a.y))*de;
      if(0 <= t && t <= 1 && 0 <= T && T <= 1) return true;
    }
  }
  return false;
}

real[][] zero(int n)
{
  real[][] m;
  for(int i=0; i < n; ++i)
    m[i]=sequence(new real(int x){return 0;},n);
  return m;
}

real[][] identity(int n)
{
  real[][] m;
  for(int i=0; i < n; ++i)
    m[i]=sequence(new real(int x){return x == i ? 1 : 0;},n);
  return m;
}

real[][] operator + (real[][] a, real[][] b)
{
  real[][] m;
  for(int i=0; i < a.length; ++i)
    m[i]=a[i]+b[i];
  return m;
}

real[][] operator - (real[][] a, real[][] b)
{
  real[][] m;
  for(int i=0; i < a.length; ++i)
    m[i]=a[i]-b[i];
  return m;
}

real[][] operator * (real[][] a, real[][] b)
{
  int n=a.length;
  real[][] m=new real[n][b[0].length];
  for(int i=0; i < n; ++i) {
    real[] ai=a[i];
    real[] mi=m[i];
    if(ai.length != b.length) 
      abort("Multiplication of incommensurate matrices is undefined");
    for(int j=0; j < b[0].length; ++j) {
      real sum;
      for(int k=0; k < b.length; ++k)
	sum += ai[k]*b[k][j];
      mi[j]=sum;
    }
  }
  return m;
}

real[][] operator * (real[][] a, real[] b)
{
  return a*transpose(new real[][] {b});
}

real[][] operator * (real[] b,real[][] a)
{
  return new real[][] {b}*a;
}

real[][] operator * (real[][] a, real b)
{
  real[][] m;
  for(int i=0; i < a.length; ++i)
    m[i]=a[i]*b;
  return m;
}

real[][] operator * (real b, real[][] a)
{
  return a*b;
}

real[][] operator / (real[][] a, real b)
{
  return a*(1/b);
}

real determinant(real[][] m)
{
  if(m.length != 2)
    abort("determinant of general matrix not yet implemented");
  if(m[0].length != m.length || m[1].length != m.length)
    abort("matrix not square");
  return m[0][0]*m[1][1]-m[0][1]*m[1][0];
}

real[][] inverse(real[][] m)
{
  if(m.length != 2)
    abort("inverse of general matrix not yet implemented");
  return new real[][] {{m[1][1],-m[0][1]},{-m[1][0],m[0][0]}}/determinant(m);
}
