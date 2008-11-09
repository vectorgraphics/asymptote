import graph;

size(0,100);

real f(real t) {return 1+cos(t);}

path g=polargraph(f,0,2pi,operator ..)--cycle;
filldraw(g,pink);

xaxis("$x$",above=true);
yaxis("$y$",above=true);

dot("$(a,0)$",(1,0),N);
dot("$(2a,0)$",(2,0),N+E);

