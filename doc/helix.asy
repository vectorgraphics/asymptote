import three;
import graph;
import graph3;
size(0,200);

currentprojection=orthographic((4,6,3));

real x(real t) {return cos(2pi*t);}
real y(real t) {return sin(2pi*t);}
real z(real t) {return t;}

draw(Label("$x$",1),(0,0,0)--(2,0,0),red,Arrow);
draw(Label("$y$",1),(0,0,0)--(0,2,0),red,Arrow);
draw(Label("$z$",1),(0,0,0)--(0,0,4),red,Arrow);

path3 p=graph(x,y,z,0,2.6);
draw(box(min(p),max(p)),blue);
draw(p,Arrow);



