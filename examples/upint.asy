import graph;
import lowupint;

size(100,0);

real a=-0.8, b=1.2;
real c=-1.0/sqrt(3.0);

partition(a,b,c,max);

arrow("$f(x)$",F(0.5*(a+b)),NNE,red);
label("$\cal{U}$",(0.5*(a+b),f(0.5*(a+b))/2));

