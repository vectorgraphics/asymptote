import graph;

size(400,200,IgnoreAspect);

real f(real t) {return cos(2pi*t);}

draw(graph(f,0.01,1));

xaxis("$x$",BottomTop,LeftTicks);
yaxis("$y$",LeftRight,RightTicks);

shipout();

