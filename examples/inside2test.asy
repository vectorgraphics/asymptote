import inside2;
//settings.outformat="pdf";

pair[] square =
  { (0,0)
  , (1,0)
  , (1,1)
  , (0,1)
  };

pair[] selfintersecting =
  { (0,0)
  , (0,1)
  , (1,0)
  , (1,1)
  };

pair[] star =
  { (0,1.65)
  , (-.5,.8)
  , (-1.75,.9)
  , (-.8,-.3)
  , (-1.2,-1.5)
  , (0,-.5)
  , (1.2,-1.5)
  , (.8,-.3)
  , (1.75,.9)
  , (.5,.8)
  };

pair[] toothbox =
  { (0,0)
  , (.7, 0)
  , (.75,.65)
  , (.9,0)
  , (1,0)
  , (1,1)
  , (0,1)
  };

pair[] horseshoe =
  { (0,0)
  , (0,1)
  , (0.4,1)
  , (0.4,0.5)
  , (0.6,0.5)
  , (0.6,1)
  , (1,1)
  , (1,0)
  };

path toPath(pair[] shape) {
  path p = shape[0];
  for (int i=1; i<shape.length; ++i) {
    p = p--shape[i];
  }
  return p--cycle;
}

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
//path g=rotate(45)*unitsquare;
//path g=rotate(45)*polygon(5);
//path g=rotate(45)*((0,0)--(0,1)--(0.4,1)--(0.4,0.5)--(0.6,0.5)--(0.6,1)--(1,1)--(1,0)--cycle);

path g=(0,0)--(1,0)--(1,1)--cycle;
import shapes;
path g=toPath(star);
path g=toPath(selfintersecting);
path g=toPath(square);
path g=toPath(toothbox);
//path g=toPath(horseshoe);
//draw(g);
pair[] p=points(g);

void test(pair z) {
  //  bool v1=inside(g,z);
  bool v2=insidePolygon(p,z);
  if(v2)
    dot(z,blue);
  else
    dot(z,red);
  //  if(v2)
  //    dot(z,red+0.375mm);
  //  assert(v1 == v2);

  //  if(insideOpt(p,z))
  //    dot(z,green+0.25mm);

  //  if(insidepolygon0(p,z) ^ inside(g,z))
  //    dot(z,red+0.375mm);
}

test((-0.1,-0.1));

for(int i=0; i < N; ++i) {
  real x=5*unitrand()-3;
  real y=5*unitrand()-3;
  test((x,y));
}

//if(false)
test(point(g,0));
for(int i=0; i < N; ++i) {
  test(point(g,0));
  test(point(g,unitrand()*length(g)));
}
test(point(g,length(g)));

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
