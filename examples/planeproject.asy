import graph3;

size3(200,IgnoreAspect);

currentprojection=orthographic(4,6,3);

real x(real t) {return 1+cos(2pi*t);}
real y(real t) {return 1+sin(2pi*t);}
real z(real t) {return t;}

path3 p=graph(x,y,z,0,1,operator ..);

draw(p,Arrow3);
draw(planeproject(XY*unitsquare3)*p,red,Arrow3);
draw(planeproject(YZ*unitsquare3)*p,green,Arrow3);
draw(planeproject(ZX*unitsquare3)*p,blue,Arrow3);

axes3("$x$","$y$","$z$");
