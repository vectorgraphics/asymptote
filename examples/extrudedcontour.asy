import contour;
import palette;
import graph3;

defaultrender.merge=true;

currentprojection=orthographic(25,10,10);
size(0,12cm);
real a=3;
real b=4;
real f(pair z) {return (z.x+z.y)/(2+cos(z.x)*sin(z.y));}
guide[][] g=contour(f,(-10,-10),(10,10),new real[]{8},150);

for(guide p:g[0]){
  draw(extrude(p,8Z),palered);
  draw(path3(p),red+2pt);
}

draw(lift(f,g),red+2pt);

surface s=surface(f,(0,0),(10,10),20,Spline);
s.colors(palette(s.map(zpart),Rainbow()+opacity(0.5)));
draw(s);
axes3("$x$","$y$","$z$",Arrow3);

