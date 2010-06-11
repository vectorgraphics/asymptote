import graph3;
import palette;
size(200);

currentprojection=orthographic(4,2,4);

triple f(pair z) {return expi(z.x,z.y);}

surface s=surface(f,(0,0),(pi,2pi),10,Spline);
draw(s,mean(palette(s.map(zpart),BWRainbow())),black,nolight,render(merge=true));
