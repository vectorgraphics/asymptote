// Riemann surface of z^{1/n}
import graph3;
import palette;

int n=3;

size(200,300,keepAspect=false);
 
currentprojection=orthographic(10,10,30);
currentlight=(10,10,5);
triple f(pair t) {return (t.x*cos(t.y),t.x*sin(t.y),t.x^(1/n)*sin(t.y/n));}
 
surface s=surface(f,(0,0),(1,2pi*n),8,16,Spline);
s.colors(palette(s.map(zpart),Rainbow()));

draw(s,meshpen=black,render(merge=true));
