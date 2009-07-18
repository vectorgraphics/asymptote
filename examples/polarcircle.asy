import math;
import graph;
size(0,100);

real f(real t) {return 2*cos(t);}
pair F(real x) {return (x,f(x));}

draw(polargraph(f,0,pi,operator ..));

defaultpen(fontsize(10pt));

xaxis("$x$");
yaxis("$y$");

real theta=radians(50);
real r=f(theta);
draw("$\theta$",arc((0,0),0.5,0,degrees(theta)),red,Arrow,PenMargins);

pair z=polar(r,theta);
draw(z--(z.x,0),dotted+red);
draw((0,0)--(z.x,0),dotted+red);
label("$r\cos\theta$",(0.5*z.x,0),0.5*S,red);
label("$r\sin\theta$",(z.x,0.5*z.y),0.5*E,red);
dot("$(x,y)$",z,N);
draw("r",(0,0)--z,0.5*unit(z)*I,blue,Arrow,DotMargin);

dot("$(a,0)$",(1,0),NE);
dot("$(2a,0)$",(2,0),NE);


