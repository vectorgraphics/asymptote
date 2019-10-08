import graph3;
import grid3;
import palette;

currentprojection=orthographic(0.8,1,1);

size(400,300,IgnoreAspect);

defaultrender.merge=true;

real f(pair z) {return cos(2*pi*z.x)*sin(2*pi*z.y);}

surface s=surface(f,(-1/2,-1/2),(1/2,1/2),50,Spline);

draw(s,mean(palette(s.map(zpart),Rainbow(40))),black);

grid3(XYZgrid);
