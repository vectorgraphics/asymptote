import graph3;
import grid3;
import palette;

currentlight=Viewport;

if(settings.render <= 0) settings.prc=false;

currentprojection=orthographic(dir(40,60));

size(400,300,IgnoreAspect);

real f(pair z) {return cos(2*pi*z.x)*sin(2*pi*z.y);}

surface s=surface(f,(-1/2,-1/2),(1/2,1/2),20,Spline);
s.colors(palette(s.map(zpart),Rainbow()));

draw(s,render(tessellate=false));

scale(true);

xaxis3(Label("$x$",0.5),Bounds,InTicks);
yaxis3(Label("$y$",0.5),Bounds,InTicks);
zaxis3(Label("$z$",0.5),Bounds,InTicks(beginlabel=false));

grid3(XYZgrid);
