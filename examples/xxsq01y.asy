import solids;
size(0,150);
currentprojection=perspective(0,0,10);
currentlight=(4,2,0);
pen color=green;
real alpha=240;

real f(real x) {return x^2;}
triple F(real x) {return (x,f(x),0);}
triple H(real x) {return (x,x,0);}

guide3 p=graph(F,0,1,n=10)--graph(H,1,0,n=10)--cycle;
revolution a=revolution(p,Y,180,180+alpha);
a.filldraw(12,color,blue);
filldraw(p,color);
filldraw(rotate(alpha,(0,1,0))*p,color);

bbox3 b=autolimits(O,1.25*(X+Y)+Z);

xaxis(Label("$x$",1),b,Arrow);
yaxis(Label("$y$",1),b,dashed,Arrow);
dot(Label("$(1,1)$"),(1,1,0));
arrow("$y=x^{2}$",F(0.7),E,0.5cm); 
arrow("$y=x$",(0.8,0.8,0),N,0.75cm); 

real r=0.4;
draw((r,f(r),0)--(r,r,0),red);
draw("$r$",(0,(f(r)+r)*0.5,0)--(r,(f(r)+r)*0.5,0),N,red,Arrows,PenMargins);
draw(arc(1.04Y,0.3,90,0,7.5,180),ArcArrow);
