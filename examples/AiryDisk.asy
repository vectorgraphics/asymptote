import graph3;
import gsl;

size(10cm,15cm,IgnoreAspect);
currentprojection=orthographic(150,50,1);

real f(pair z) {real r=abs(z); return r == 0 ? 1 : (2.0*J(1,r)/r)^2;}

real R=15;
surface s=surface(f,(-R,-R),(R,R),100,Spline);

draw(s,green);

xaxis3("$x$",Bounds,InTicks);
yaxis3("$y$",Bounds,InTicks);
zaxis3(rotate(90)*"$I(\sqrt{x^2+y^2})$",Bounds,InTicks("$%#.1f$"));
