import math;
import graph;
size(0,100);

real f(real t) {return 2*cos(t);}
pair F(real x) {return (x,f(x));}

draw(polargraph(f,0,pi,Spline));

currentpen=fontsize(8);
pen Red=currentpen+red;

xaxis("$x$");
yaxis("$y$");

real theta=radians(50);
real r=f(theta);
draw("$\theta$",arc((0,0),0.5,0,degrees(theta)),Red,Arrow);

pair z=polar(r,theta);
draw(z--(z.x,0),dotted+Red);
draw((0,0)--(z.x,0),dotted+Red);
label("$r\cos\theta$",(0.5*z.x,0),0.5*S,Red);
label("$r\sin\theta$",(z.x,0.5*z.y),0.5*E,Red);
labeldot("$(x,y)$",z,N);
draw("r",(0,0)--z,0.5*unit(z)*I,Red,Arrow);

labeldot("$(a,0)$",(1,0),NE);
labeldot("$(2a,0)$",(2,0),NE);

shipout();

