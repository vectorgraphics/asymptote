import graph;

size(0,100);

real f(real t) {return 1+cos(t);}

path g=polargraph(f,0,2pi,operator ..)--cycle;
filldraw(g,pink);

xaxis("$x$",Above);
yaxis("$y$",Above);

dot("$(a,0)$",(1,0),N);
dot("$(2a,0)$",(2,0),N+E);

