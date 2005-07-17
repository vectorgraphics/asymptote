import graph;
size(0,100);

real f(real x) {return sqrt(x);}
pair F(real x) {return (x,f(x));}

real g(real x) {return sqrt(-x);}
pair G(real x) {return (x,g(x));}

guide p=graph(f,0,1,Spline);

draw(p--(0,1),dotted);

guide q=(0,0)--p--(0,1);
guide h=q--cycle;
fill(h,lightgray);

xaxis("$x$");
yaxis(0,1.25,"$y$",dashed);

draw(q);

arrow("$y=\sqrt{x}$",F(0.7),ESE);

draw(graph(g,0,-1,Spline)--(0,1),dotted);

draw((0,0.74),yscale(0.5)*arc(0,0.25cm,-250,70),ArcArrow);

labelxtick(-1);
labelx(0);
labelxtick(1);

labeldot((1,1));

