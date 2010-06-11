import graph3;
import palette;
       
size(200,300,keepAspect=false);
//settings.nothin=true;
       
currentprojection=orthographic(10,10,30);
currentlight=(10,10,5);
triple f(pair t) {return (exp(t.x)*cos(t.y),exp(t.x)*sin(t.y),t.y);}
       
surface s=surface(f,(-4,-2pi),(0,4pi),8,16,Spline);
s.colors(palette(s.map(zpart),Rainbow()));
draw(s,render(merge=true));
