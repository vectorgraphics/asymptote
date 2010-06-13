import graph3;
import contour;

currentprojection=orthographic(1,-2,1);
currentlight=White;

size(12cm,0);

real sinc(pair z) {
  real r=2pi*abs(z);
  return r != 0 ? sin(r)/r : 1;
}

render render=render(compression=Low,merge=true);

draw(lift(sinc,contour(sinc,(-2,-2),(2,2),new real[] {0})),red);
draw(surface(sinc,(-2,-2),(2,2),Spline),lightgray,render);

draw(scale3(2*sqrt(2))*unitdisk,paleyellow+opacity(0.25),nolight,render);
draw(scale3(2*sqrt(2))*unitcircle3,red,render);

xaxis3("$x$",Bounds,InTicks);
yaxis3("$y$",Bounds,InTicks(beginlabel=false));
zaxis3("$z$",Bounds,InTicks);
