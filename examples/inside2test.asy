import inside2;
settings.outformat="pdf";

struct StraightContribution {
  pair outside;
  int count=0;

  void operator init(pair outside)  {
    this.outside=outside;
  }

  bool onBoundary(pair z0, pair z1, pair z) {
    int s1 = sgn(orient(z,z0,z1));
    int s2 = sgn(orient(outside,z0,z1));

    if (s1 == s2 && s1 != 0)
      return false;

    int s3 = sgn(orient(z,outside,z0));
    int s4 = sgn(orient(z,outside,z1));
    if (s3 != s4) {
      if (s1 == 0) return true;
      count += s3;
    } else {
      if (s1 != 0) return false;
      // only do these checks if outside is not colinear with z0 z1
      //    if (s2 != 0) return false; // CHECK
      if (z0 == z1)
        return z == z0;
      if (s3 == 0 && s4 == 0) {
        real norm = sqrt(max(abs2(z0), abs2(z1), abs2(z)));
        do {
          pair ref = (unitrand()*norm, unitrand()*norm);
          s3=sgn(orient(z,ref,z0));
          s4=sgn(orient(z,ref,z1));
        } while (s3 == 0 && s4 == 0);
      }
      return s3 != s4;
    }
    return false;
  }
}

bool insidepolygon(pair[] polygon, pair p) {
  pair outside = 2*maxbound(polygon) - minbound(polygon);
  var W=StraightContribution(outside);
  pair prevPoint = polygon[polygon.length - 1];
  for (int i=0; i < polygon.length; ++i) {
    pair currentPoint = polygon[i];
    if(W.onBoundary(prevPoint,currentPoint,p)) return true;
    prevPoint = currentPoint;
  }
  return W.count != 0;
}

bool inrange(real x0, real x1, real x) {
  return (x0 <= x && x <= x1) || (x1 <= x && x <= x0);
}


// Returns a list of nodes for a given path p
pair[] points(path p)
{
  int n=size(p);
  pair[] v;
  for(int i=0; i < n; ++i)
    v.push(point(p,i));
  return v;
}


int count=0;

size(10cm);

int N=1000;
path g=unitsquare;
//path g=rotate(45)*polygon(5);
draw(g);
pair[] p=points(g);

void test(pair z) {
  if(inside(g,z))
    dot(z,blue+opacity(0.5)+0.5mm);
  if(insidepolygon(p,z))
    dot(z,red+0.375mm);

  //  if(insideOpt(p,z))
  //    dot(z,green+0.25mm);

  //  if(insidepolygon0(p,z) ^ inside(g,z))
  //    dot(z,red+0.375mm);
}

for(int i=0; i < N; ++i) {
  real x=5*unitrand()-3;
  real y=5*unitrand()-3;
  test((x,y));
  }

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
