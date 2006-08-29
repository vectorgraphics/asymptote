import graph3;
import solids;
size(0,150);
currentprojection=perspective(0,0,10);
currentlight=(1,0,0.25);

real f(real x) {return sqrt(x);}
triple F(real x) {return (x,f(x),0);}

guide3 p=graph(F,0,1,n=25);
revolution a=revolution(p,Y,0,360);
a.filldraw(20,green,blue,false);
draw(p,blue);

bbox3 b=autolimits(O,1.25*(X+Y)+Z);

xtick((0,0,0));
xtick((1,0,0));

xaxis(Label("$x$",1),b,Arrow);
yaxis(Label("$y$",1),b,dashed,Arrow);
dot(Label("$(1,1)$"),(1,1,0));
arrow("$y=\sqrt{x}$",F(0.5),E,0.75cm);
draw(reverse(arc(1.01Y,0.3,90,0,7.5,180)),ArcArrow);
