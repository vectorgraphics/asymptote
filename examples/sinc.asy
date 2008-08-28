import graph3;
import contour;

currentprojection=orthographic(1,-2,1);
currentlight=(1,-1,0.5);

size(12cm,0);

real sinc(pair z) {
  real r=2pi*abs(z);
  return r != 0 ? sin(r)/r : 1;
}

draw(lift(sinc,contour(sinc,(-2,-2),(2,2),new real[] {0})));
draw(surface(sinc,(-2,-2),(2,2),Spline),lightgray+opacity(0.5));

xaxis3("$x$",Bounds,LeftTicks3(beginlabel=false,Label));
yaxis3("$y$",Bounds,LeftTicks3(Label));
zaxis3("$z$",Bounds,LeftTicks3());
