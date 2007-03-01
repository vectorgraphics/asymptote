import graph3;
import contour;

size(12cm,0);

real sinc(pair z) {
  real r=2pi*abs(z);
  return r != 0 ? sin(r)/r : 1;
}

bbox3 b=limits((-2,-2,-0.2),(2,2,1.2));
currentprojection=orthographic(1,-2,1);
currentlight=(1,-1,0.5);

aspect(b,1,1,1);

xaxis(rotate(X)*"$x$",b,RightTicks(rotate(X)*Label));
yaxis(rotate(Y)*"$y$",b.X(),b.XY(),LeftTicks(rotate(Y)*Label));
zaxis("$z$",b,RightTicks());

layer();

draw(lift(sinc,contour(sinc,(-2,-2),(2,2),new real[] {0})));
add(surface(sinc,xypart(b.O()),xypart(b.XY()),50,lightgray+opacity(0.5)));
