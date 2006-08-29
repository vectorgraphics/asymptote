import graph3;
import solids;
size(0,150);
currentprojection=orthographic(1.5,0,10,up=Y);
pen color=green;

real f(real x){return sqrt(x);}
triple F(real x){return (x,f(x),0);}

guide3 p=graph(F,0,1,n=20);
revolution a=revolution(p,X,0,360);
a.filldraw(16,color,3,blue);
draw(p,blue);

bbox3 b=autolimits(O,2X+1.25Y+Z);

xtick((0,0,0));
xtick((1,0,0));

real x=relpoint(p,0.5).x;

xaxis(Label("$x$",1),b,dashed,Arrow);
yaxis(Label("$y$",1),b,Arrow);
dot(Label("$(1,1)$"),(1,1,0));
arrow(Label("$y=\sqrt{x}$"),F(0.7),dir(75),0.6cm);
draw(arc(1.43X,0.4,90,90,175,-40,CW),ArcArrow);
draw("$r$",(x,0,0)--(x,f(x),0),red,Arrow,PenMargin);
