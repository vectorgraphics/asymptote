import math;
import graph;
size(0,100);

real d=Tan(30);
pair z1=(-1,-d);
pair z2=-z1;
filldraw(z1--z2--(1,0)--(-1,0)--cycle,red);

xaxis(red,"$x$");
yaxis(dotted,"$y$");

draw(z1--2*z1,red);
draw(z2--2*z2,red);

xequals(-1,-2,2);
xequals(1,-2,2);

