import three;

//settings.outformat="pdf";

struct StraightContribution {
  triple outside,h;
  int count=0;

  void operator init(triple outside, triple h)  {
    this.outside=outside;
    this.h=h;
  }

  // given 3 colinear points, return true if point v lies between points v0 and v1.
  bool insideSegment(triple z0, triple z1, triple z) {
    if(z == z1 || z == z0) return true;
    if(z0 == z1) return false;

    triple crossproduct=cross(z0-z,z1-z);
    triple H=z+crossproduct;
    int s1 = sgn(orient(z0,z,H,h));
    int s2 = sgn(orient(z1,z,H,h));
    return s1 != s2;
  }

  bool onBoundary(triple z0, triple z1, triple z) {
    int s1 = sgn(orient(z,z0,z1,h));
    int s2 = sgn(orient(outside,z0,z1,h));

    if (s1 == s2 && s1 != 0)
      return false;

    int s3 = sgn(orient(z,outside,z0,h));
    int s4 = sgn(orient(z,outside,z1,h));
    if (s3 != s4) {
      if (s1 == 0) return true;
      count += s3;
    } else if (s1 == 0)
      return insideSegment(z0,z1,z);
    return false;
  }
}

// return true if v lies within planar polygon p
bool insidePolygon(triple[] p, triple v) {
  triple prevPoint = p[p.length - 1];

  triple outside = 2*maxbound(p) - minbound(p);

  triple crossproduct=cross(prevPoint-v,p[0]-v);
  triple h=v+crossproduct;
  triple n=unit(crossproduct);
  outside=outside-dot(outside,n)*n;

  var W=StraightContribution(outside,h);
  for (int i=0; i < p.length; ++i) {
    triple currentPoint = p[i];
    if(W.onBoundary(prevPoint,currentPoint,v)) return true;
    prevPoint = currentPoint;
  }
  return W.count != 0;
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

int N=100000;
path3 g=rotate(45,X)*rotate(30,Y)*path3(polygon(11));
//path g=rotate(45)*polygon(5);
draw(g);
triple[] p=points(g);

void test(triple z) {
  //  if(inside(g,z))
  //    dot(z,blue+opacity(0.5)+0.5mm);
  if(insidePolygon(p,z))
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
  }

/*
for(int i=0; i < N; ++i) {
  real x=3*unitrand()-2;
  test((x,0));
  test((0,x));
  test((x,1));
  test((1,x));
  test((x,-1e-99));
  test((-1e-99,x));
  test((x,1e-99));
  test((1e-99,x));
  test((x,-1e-16));
  test((-1e-16,x));
  test((x,1e-16));
  test((1e-16,x));
  test((x,-3e-16));
  test((-3e-16,x));
  test((x,3e-16));
  test((3e-16,x));
  test((x,-1e-8));
  test((-1e-8,x));
  test((x,1e-8));
  test((1e-8,x));
}
*/
