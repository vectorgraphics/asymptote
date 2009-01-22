size(200,IgnoreAspect); 
import graph; 
real f(real x) {return 1/x;}; 
draw(graph(f,-1,1,new bool(real x) {return x != 0;})); 
axes("$x$","$y$",red);
