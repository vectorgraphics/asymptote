import three;
import graph;
import graph3;
size(0,200);

currentprojection=orthographic((4,4,3));

real x(real t) {return cos(2pi*t);}
real y(real t) {return sin(2pi*t);}
real z(real t) {return t;}

draw(graph(x,y,z,0,3),Arrow);

draw("$x$",(0,0,0)--(2,0,0),1,red,Arrow);
draw("$y$",(0,0,0)--(0,2,0),1,red,Arrow);
draw("$z$",(0,0,0)--(0,0,4),1,red,Arrow);


