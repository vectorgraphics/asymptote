import graph;

size(0,200);

real f(real t) {return cos(2pi*t);}
real g(real t) {return sin(2pi*t);}

draw(graph(f,g,0,1,LinearInterp));
//limits((0,-1),(1,0));

xaxis("$x$",BottomTop,LeftTicks);
yaxis("$E(k)$",LeftRight,RightTicks);

shipout();

