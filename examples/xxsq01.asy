import graph3;
import solids;
size(0,150);
currentprojection=perspective(0,0,10);
currentlight=(1,2,-1);
pen color=green;
real alpha=250;

real f(real x) {return x^2;}
triple F(real x) {return (x,f(x),0);}
triple H(real x) {return (x,x,0);}

path3 p=graph(F,0,1,n=10)--graph(H,1,0,n=10)--cycle;
revolution a=revolution(p,X,-alpha,0);
a.filldraw(8,color,blue,false);
filldraw(p,color);
filldraw(rotate(-alpha,X)*p,color);
draw(p,blue);

bbox3 b=autolimits(O,1.6X+1.25*Y+Z);

xaxis(Label("$x$",1),b,dashed,Arrow);
yaxis(Label("$y$",1),b,Arrow);
dot(Label("$(1,1)$"),(1,1,0));
arrow("$y=x$",(0.7,0.7,0),N,0.75cm);
arrow("$y=x^{2}$",F(0.7),E,0.75cm);
draw(arc(1.25X,0.3,90,90,3,-90),ArcArrow);
