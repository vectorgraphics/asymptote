
import graph;
size(150,0);

real f(real x) {return exp(x);}
pair F(real x) {return (x,f(x));}

xaxis("$x$");
yaxis(0,"$y$");

draw(graph(f,-4,2,Spline),red);

ylabel(1,E);
label("$e^x$",F(1),SE);

