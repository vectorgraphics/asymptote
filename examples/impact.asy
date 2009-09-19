// Contributed by Philippe Ivaldi.
// http://www.piprime.fr/

import graph3 ;
import contour;
size (6cm,0);
currentprojection=orthographic(1,1,1) ;

real rc=1, hc=2, c=rc/hc;
draw(shift(hc*Z)*scale(rc,rc,-hc)*unitcone,blue);

triple Os=(0.5,0.5,1);
real r=0.5;
draw(shift(Os)*scale3(r)*unitsphere,red);

real a=1+1/c^2;
real b=abs(Os)^2-r^2;

real f(pair z)
{
  real x=z.x, y=z.y;
  return a*x^2-2*Os.x*x+a*y^2-2*Os.y*y-2*Os.z*sqrt(x^2+y^2)/c+b;
}

real g(pair z){return (sqrt(z.x^2+z.y^2))/c;}

draw(lift(g,contour(f,(-rc,-rc),(rc,rc),new real[]{0})),linewidth(2bp)+yellow);

axes3("$x$","$y$","$z$",Arrow3);
