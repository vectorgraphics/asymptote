size(0,150);
import graph;

real f(real x) {return sqrt(x);}
pair F(real x) {return (x,f(x));}

real g(real x) {return -sqrt(x);}
pair G(real x) {return (x,g(x));}

guide p=(0,0)--graph(f,0,1,operator ..)--(1,0);
fill(p--cycle,lightgray);
draw(p);
draw((0,0)--graph(g,0,1,operator ..)--(1,0),dotted);

real x=0.5;
pair c=(4,0);

transform T=xscale(0.5);
draw((2.75,0),T*arc(0,0.30cm,20,340),ArcArrow);
fill(shift(c)*T*circle(0,-f(x)),red+white);
draw(F(x)--c+(0,f(x)),dashed+red);
draw(G(x)--c+(0,g(x)),dashed+red);

dot(Label,(1,1));
arrow("$y=\sqrt{x}$",F(0.7),N);

arrow((3,0.5*f(x)),W,1cm,red);
arrow((3,-0.5*f(x)),W,1cm,red);

xaxis(0,c.x,"$x$",dashed);
yaxis("$y$");

draw("$r$",(x,0)--F(x),E,red,Arrows,BeginBar,PenMargins);
draw("$r$",(x,0)--G(x),E,red,Arrows,PenMargins);
draw("$r$",c--c+(0,f(x)),Arrow,PenMargin);
dot(c);
