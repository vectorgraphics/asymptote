import graph3;
import solids;
size(0,150);
currentprojection=perspective(0,0,10);
currentlight=(1,1,0);
pen color=green;
real alpha=240;

real f(real x) {return sqrt(x);}
triple F(real x) {return (x,f(x),0);}

guide3 p=graph(F,0,1,n=30);
revolution a=revolution(p,X,180,180+alpha);
a.filldraw(color,blue,false);
p=p--X--cycle3;
filldraw(p,color);
filldraw(rotate(alpha,X)*p,color);

bbox3 b=autolimits(O,1.7X+1.5*Y+Z);

xaxis(Label("$x$",1),b,dashed,Arrow);
yaxis(Label("$y$",1),b,Arrow);
dot("$(1,1)$",(1,1,0));
arrow("$y=\sqrt{x}$",F(0.8),N,0.75cm);

real r=0.4;
draw(F(r)--(1,f(r),0),red);
real x=(1+r)/2;

draw("$r$",(x,0,0)--(x,f(r),0),red,Arrow);
draw(arc(1.4X,0.4,90,90,3,-90),ArcArrow);
