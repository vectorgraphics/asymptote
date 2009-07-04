import graph3;

size(0,200);
size3(200,IgnoreAspect);

currentprojection=orthographic(4,6,3);

real x(real t) {return cos(2pi*t);}
real y(real t) {return sin(2pi*t);}
real z(real t) {return t;}

path3 p=graph(x,y,z,0,2.7,operator ..);

draw(p,Arrow3);

scale(true);

xaxis3(XZ()*"$x$",Bounds,red,InTicks(Label,2,2));
yaxis3(YZ()*"$y$",Bounds,red,InTicks(beginlabel=false,Label,2,2));
zaxis3(XZ()*"$z$",Bounds,red,InTicks);
