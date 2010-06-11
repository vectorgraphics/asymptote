import graph3;
import palette;
 
size3(10cm);
 
currentprojection=orthographic(5,4,2);
viewportmargin=(2cm,0);

real r(real t) {return 3exp(-0.1*t);}
real x(real t) {return r(t)*cos(t);}
real y(real t) {return r(t)*sin(t);}
real z(real t) {return t;}

path3 p=graph(x,y,z,0,6*pi,50,operator ..);

tube T=tube(p,2);
surface s=T.s;
s.colors(palette(s.map(zpart),BWRainbow()));
draw(s,render(merge=true));
draw(T.center,thin());
