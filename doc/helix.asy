import graph3;

size(0,200);
size3(200,IgnoreAspect);

currentprojection=orthographic(4,6,3);

real x(real t) {return cos(2pi*t);}
real y(real t) {return sin(2pi*t);}
real z(real t) {return t;}

defaultpen(overwrite(SuppressQuiet));

path3 p=graph(x,y,z,0,2.7,operator ..);

//draw(p,Arrow);
draw(p);

xaxis3(XY()*"$x$",Bounds(),red,LeftTicks3(Label,2,2));
yaxis3(YX()*"$y$",Bounds(),red,LeftTicks3(Label,2,2));
zaxis3(XZ()*"$z$",Bounds(),red,LeftTicks3);


