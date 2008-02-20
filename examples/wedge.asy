import graph3;
import solids;
size(0,150);
currentprojection=perspective(8,10,2);
pen color=green;

revolution r=cylinder(-4Z,4,8,Z);
draw(circle(O,4,Z));
r.draw(color);

triple F(real x){return (x,sqrt(16-x^2),sqrt((16-x^2)/3));}
path3 p=graph(F,0,4,operator ..)--O;
path3 q=rotate(180,(0,4,4/sqrt(3)))*p--O;
draw(p); fill(p--cycle,color);
draw(q); fill(q--cycle,color);

real t=2;
path3 triangle=(t,0,0)--(t,sqrt(16-t^2),0)--F(t)--cycle;
filldraw(triangle,red);

bbox3 b=autolimits(O,6*(X+Y+Z));

xaxis(Label("$x$",1),b,Arrow);
yaxis(Label("$y$",1),b,Arrow);
zaxis(Label("$z$",1),b,dashed,Arrow);


