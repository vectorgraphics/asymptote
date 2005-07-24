
import graph;

size(0,200);

real x(real t) {return cos(2pi*t);}
real y(real t) {return sin(2pi*t);}

draw(graph(x,y,0,1));

//xlimits(0,1);
//ylimits(-1,0);

xaxis("$x$",BottomTop,LeftTicks("$%#.1f$"));
yaxis("$y$",LeftRight,RightTicks("$%#.1f$"));


