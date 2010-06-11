import graph3;
import contour3;

size(200,0);
currentprojection=orthographic((6,8,2),up=Y);

real a(real z) {return (z < 6) ? 1 : exp((abs(z)-6)/4);}
real b(real z) {return 1/a(z);}
real B(real z) {return 1-0.5cos(pi*z/10);}

real f(real x, real y, real z) {return 0.5B(z)*(a(z)*x^2+b(z)*y^2)-1;}

draw(surface(contour3(f,(-2,-2,-10),(2,2,10),10)),blue+opacity(0.75),
     render(merge=true));

xaxis3(Label("$x$",1),red);
yaxis3(Label("$y$",1),red);
zaxis3(Label("$z$",1),red);
