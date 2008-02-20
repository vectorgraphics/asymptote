import graph3;
import solids;
size(0,150);
currentprojection=orthographic(0,1.25,10,up=Y);
currentlight=(5,2,-1);
pen color=green;

real f(real x) {return x^2;}
triple F(real x) {return (x,f(x),0);}
triple H(real x) {return (x,x,0);}

path3 p=graph(F,0,1,n=10)--graph(H,1,0,n=10)--cycle;
revolution a=revolution(-X,p,Y,180,360);
a.filldraw(16,color,blue,false);
filldraw(p,color);
filldraw(shift(-2X)*rotate(180,Y)*p,color);

bbox3 b=autolimits(O,1.25*(X+Y)+Z);

draw((-1,0,0)--(-1,1,0),dashed);
xaxis(Label("$x$",1),b,Arrow);
yaxis(Label("$y$",1),b,Arrow);
dot(Label("$(1,1)$"),(1,1,0));
dot(Label("$(-1,1)$"),(-1,1,0),W);
arrow("$y=x^{2}$",F(0.7),E,0.5cm); 
arrow("$y=x$",(0.3,0.3,0),N,0.5cm); 
draw(circle((-1,1,0),2,Y),dashed);
draw((-1,1,0)--(1,1,0),dashed);
draw(shift(-X)*arc(0.02Y,0.3,90,0,0,0,CW),ArcArrow);

