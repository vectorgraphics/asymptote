import graph;
size(0,100);

real f(real t) {return cos(2*t);}

path g=polargraph(f,0,2pi,operator ..)--cycle;
fill(g,green+white);
xaxis("$x$",Above);
yaxis("$y$",Above);
draw(g);

dot(Label,(1,0),NE);
dot(Label,(0,1),NE);


