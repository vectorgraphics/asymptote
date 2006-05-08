import graph;
size(0,100);

real d=Tan(30);
pair z1=(-1,-d);
pair z2=-z1;
filldraw(z1--z2--(1,0)--(-1,0)--cycle,red);

xaxis("$x$",red);
yaxis("$y$",dotted);

draw(z1--2*z1,red);
draw(z2--2*z2,red);

yaxis(XEquals(1),-2,2);
yaxis(XEquals(-1),-2,2);

