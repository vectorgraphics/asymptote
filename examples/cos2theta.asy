import graph;
size(0,100);

real f(real t) {return cos(2*t);}

guide g=polargraph(f,0,2pi,Spline)--cycle;
fill(g,green+white);
xaxis("$x$");
yaxis("$y$");
draw(g);

labeldot((1,0),NE);
labeldot((0,1),NE);

shipout();

