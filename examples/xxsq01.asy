import graph;
size(0,125);

real f(real x) {return x^2;}
pair F(real x) {return (x,f(x));}

real g(real x) {return x;}
pair G(real x) {return (x,g(x));}

guide h=(0,0)--graph(g,0,1);

arrow("$y=x^2$",F(0.85),E,0.75cm);
arrow("$y=x$",G(0.4),N);

guide g=(0,0)--graph(f,0,1,operator ..)--cycle;
fill(g,lightgray);
draw(h);
draw(g);

transform T=xscale(0.5);
draw((1.04,0),T*arc(0,0.25cm,20,340),ArcArrow);

dot(Label,(1,1));

real x=0.5;
pair c=(3,0);
picture canvas=new picture; 
transform S0=shift(c)*T*shift(-c);
fill(canvas,S0*circle(c,g(x)),red+white);
unfill(canvas,S0*circle(c,f(x)));
add(currentpicture,canvas);

draw((x,g(x))--c+(0,g(x)),dashed+red);
draw((x,f(x))--c+(0,f(x)),dashed+red);
draw((x,f(x))--(x,g(x)),red);

arrow((2,0.5*(f(x)+g(x))),W,1cm,red);

xaxis(0,c.x,"$x$",dashed);
yaxis("$y$");

draw("$r_{\rm in}$",S0*(c--c+f(x)*dir(-45)),dir(56),darkgreen,Arrow(6.0));
draw("$r_{\rm out}$",S0*(c--c+(0,g(x))),blue,Arrow);
dot(c);
