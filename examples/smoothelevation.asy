import graph3;
import grid3;
import palette;

currentprojection=orthographic(0.8,1,1);

settings.prc=false;

size(400,300,IgnoreAspect);

real f(pair z) {return cos(2*pi*z.x)*sin(2*pi*z.y);}

surface s=surface(f,(-1/2,-1/2),(1/2,1/2),20,Spline);
s.colors(palette(s.cornermap(zpart),Rainbow()));

draw(s);
grid3(XYZgrid);
