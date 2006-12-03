import graph;
defaultpen(1.0);

size(0,150,IgnoreAspect);

real arrowsize=4mm;
real arrowlength=2arrowsize;

// Return a vector interpolated linearly between a and b.
vector vector(pair a, pair b) {
  return new path(real x) {
    return (0,0)--arrowlength*interp(a,b,x);
  };
}

real alpha=1;
real f(real x) {return alpha/x;}

real epsilon=0.5;
path p=graph(f,epsilon,1/epsilon);

int n=2;
draw(p);
xaxis("$x$");
yaxis("$y$");

vectorfield(p,n,vector(W,W),arrowsize);
vectorfield((0,0)--(point(E).x,0),n,vector(NE,NW),arrowsize);
vectorfield((0,0)--(0,point(N).y),n,vector(NE,NE),arrowsize);

