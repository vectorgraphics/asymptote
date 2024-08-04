import inside3;

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

int N=1000;//1000;
//path3 g=path3(polygon(5));
path3 g=unitsquare3;
transform3 t=rotate(45,Z)*rotate(30,Y);
//transform3 t=identity4;
path3 g=(0,0,0)--(1,0,0)--(1,1,0)--cycle;

path3 G=t*g;
draw(G);
triple[] p=points(G);

void test(triple Z) {
  //  if(inside(g,z))
  //    dot(z,blue+opacity(0.5)+0.5mm);
  triple z=t*Z;
  if(insidePolygon(p,z))
    dot(z,blue+opacity(0.5)+0.5mm);
  else
    dot(z,red+0.375mm);

  //  if(insideOpt(p,z))
  //    dot(z,green+0.25mm);

  //  if(insidepolygon0(p,z) ^ inside(g,z))
  //    dot(z,red+0.375mm);
}

for(int i=0; i < N; ++i) {
  triple m=min(g);
  triple M=max(g);
  real x=unitrand();
  real y=unitrand();
  triple d=M-m;
  test(m-d+3*(d.x*x,d.y*y,0));
}

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

test((-0.1,-0.1,0));

//import graph3;
//axes3(Arrow3);
