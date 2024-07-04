import three;

//settings.outformat="pdf";

struct StraightContribution {
  triple outside,normal;
  triple H;
  int count=0;

  void operator init(triple outside, triple normal)  {
    this.outside=outside;
    this.normal=normal;
  }

  // given 3 colinear points, return true if point v lies between points v0 and v1.
  bool insideSegment(triple z0, triple z1, triple z) {
    if(z == z1 || z == z0) return true;
    if(z0 == z1) return false;

    triple h = cross(z1-z0,normal);
    int s1 = sgn(orient(z0,z,h,H));
    int s2 = sgn(orient(z1,z,h,H));
    assert(s1 != 0 && s2 != 0);
    return s1 != s2;
  }

  bool onBoundary(triple z0, triple z1, triple z) {
    H=z+normal;
    int s1 = sgn(orient(z,z0,z1,H));
    int s2 = sgn(orient(outside,z0,z1,H));

    if (s1 == s2 && s1 != 0)
      return false;

    int s3 = sgn(orient(z,outside,z0,H));
    int s4 = sgn(orient(z,outside,z1,H));
    if (s3 != s4) {
      if (s1 == 0) return true;
      count += s3;
    } else if (s1 == 0)
      return insideSegment(z0,z1,z);
    return false;
  }
}

// Return the winding number of planar polygon relative to point v
// lying in the same plane, or the largest odd integer if v lies on p.
int windingnumberPolygon(triple[] p, triple v) {
  triple prevPoint = p[p.length - 1];
  triple outside = 2*maxbound(p) - minbound(p);
  real norm2=abs2(p[0]);
  for(int i=1; i < p.length; ++i)
    norm2=max(norm2,abs2(p[i]));

  triple n=normal(p);
  triple normal=sqrt(norm2)*n;
  outside=outside-dot(outside,n)*n;

  var W=StraightContribution(outside,normal);
  for (int i=0; i < p.length; ++i) {
    triple currentPoint = p[i];
    if(W.onBoundary(prevPoint,currentPoint,v)) return undefined;
    prevPoint = currentPoint;
  }
  return W.count;
}

// Return true if point v lies on or inside planar polygon p.
bool insidePolygon(triple[] p, triple v) {
  return windingnumberPolygon(p,v) != 0;
}

// Returns a list of nodes for a given path p
triple[] points(path3 p)
{
  int n=size(p);
  triple[] v;
  for(int i=0; i < n; ++i)
    v.push(point(p,i));
  return v;
}


int count=0;

size(10cm);

int N=10000;
path3 g=unitsquare3;
path3 g=rotate(45,X)*rotate(30,Y)*path3(polygon(11));
draw(g);
triple[] p=points(g);

void test(triple z) {
  //  if(inside(g,z))
  //    dot(z,blue+opacity(0.5)+0.5mm);
  if(insidePolygon(p,z))
    dot(z,blue+opacity(0.1)+0.5mm);
  else
    dot(z,red+0.375mm);

  //  if(insideOpt(p,z))
  //    dot(z,green+0.25mm);

  //  if(insidepolygon0(p,z) ^ inside(g,z))
  //    dot(z,red+0.375mm);
}

for(int i=0; i < N; ++i) {
  triple v0=point(g,0);
  triple u=point(g,1)-v0;
  triple v=point(g,2)-v0;
  test(v0+(50*unitrand()-25)*u+(50*unitrand()-25)*v);
  //  test(v0+(3*unitrand()-2)*u+(3*unitrand()-2)*v);
  }

if(false)
for(int i=0; i < N; ++i) {
  real x=3*unitrand()-2;
  test((x,0,0));
  test((0,x,0));
  test((x,1,0));
  test((1,x,0));
  //  test((x,-1e-99,0));
  //  test((-1e-99,x,0));
  test((x,1e-99,0));
  test((1e-99,x,0));
  //  test((x,-1e-16,0));
  //  test((-1e-16,x,0));
  test((x,1e-16,0));
  test((1e-16,x,0));
  //  test((x,-3e-16,0));
  //  test((-3e-16,x,0));
  test((x,3e-16,0));
  test((3e-16,x,0));
  //  test((x,-1e-8,0));
  //  test((-1e-8,x,0));
  test((x,1e-8,0));
  test((1e-8,x,0));
}

//import graph3;
//axes3(Arrow3);
