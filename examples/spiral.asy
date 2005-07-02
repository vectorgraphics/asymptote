size(0,150);
import graph;

real f(real t) {return exp(-t/(2pi));}

draw(polargraph(f,0,20*pi,Spline));

xaxis(-infinity,1.3,"$x$");
yaxis(-infinity,1,"$y$");

xlabel(1);
xlabel("$e^{-1}$",1.0/exp(1),SE);
