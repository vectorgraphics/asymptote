import graph3;
import grid3;
import palette;

currentprojection=orthographic(1,2,10);
settings.prc=false;

size(400,300,IgnoreAspect);

real f(pair z) {return cos(2*pi*z.x)*sin(2*pi*z.y);}

surface s=surface(f,(-1/2,-1/2),(1/2,1/2),20,Spline);
s.colors(palette(s.map(zpart),Rainbow()));

draw(s);
grid3(XYZgrid);

xaxis3(Label("$x$",0.5),Bounds,InTicks());
yaxis3(Label("$y$",0.5),Bounds,InTicks());
zaxis3(Label("$z$",0.5),Bounds,InTicks(beginlabel=false));
