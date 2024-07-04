import inside2;

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

  /*
  if(inside(g,z))
    dot(z,blue+opacity(0.5)+0.5mm);
  if(insidePolygon(p,z))
    dot(z,red+0.375mm);
  */

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
  /*
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
  */
}
