size(0,150);
import graph;

real f(real t) {return exp(-t/(2pi));}

draw(polargraph(f,0,20*pi,operator ..));

xaxis("$x$",-infinity,1.3);
yaxis("$y$",-infinity,1);

labelx(1);
labelx("$e^{-1}$",1.0/exp(1),SE);
