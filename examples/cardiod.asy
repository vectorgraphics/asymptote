import graph;

size(0,100);

real f(real t) {return 1+cos(t);}

guide g=polargraph(f,0,2pi,Spline)--cycle;
filldraw(g,pink+white);

xaxis(true,"$x$");
yaxis(true,"$y$");

labeldot("$(a,0)$",(1,0),N);
labeldot("$(2a,0)$",(2,0),N+E);

