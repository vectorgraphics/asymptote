import graph;

real f(real x) {return sqrt(x);}
pair F(real x) {return (x,f(x));}

real g(real x) {return -sqrt(x);}
pair G(real x) {return (x,g(x));}

guide p=(0,0)--graph(f,0,1,Spline)--(1,0);
guide h=p--cycle;

picture canvas=new picture;
fill(canvas,h,gray);
draw(canvas,p);
draw(canvas,(0,0)--graph(g,0,1,Spline)--(1,0),dotted);
add(canvas);

real x=0.5;
pair c=(4,0);
draw(F(x)--c+(0,f(x)),dashed+red);
draw(G(x)--c+(0,g(x)),dashed+red);

arrow("$y=\sqrt{x}$",F(0.7),N);

transform T=xscale(0.5);
drawabout((2.75,0),T*arc(0,0.30cm,20,340),ArcArrow);

labeldot((1,1));

fill(shift(c)*T*circle(0,-f(x)),red+white);
arrow((3,0.5*f(x)),W,1cm,red);
arrow((3,-0.5*f(x)),W,1cm,red);

xaxis(0,c.x,dashed,"$x$");
yaxis("$y$");

draw("$r$",(x,0)--F(x),E,red,Arrows,BeginBar);
draw("$r$",(x,0)--G(x),E,red,Arrows,BeginBar);
draw("$r$",c--c+(0,f(x)),Arrow);
dot(c);

shipout(0,150);
