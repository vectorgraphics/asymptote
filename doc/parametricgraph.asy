
import graph;

size(0,200);

real f(real t) {return cos(2pi*t);}
real g(real t) {return sin(2pi*t);}

draw(graph(f,g,0,1,LinearInterp));

//xlimits(0,1);
//ylimits(-1,0);

xaxis("$x$",BottomTop,LeftTicks("%#.1f"));
yaxis("$y$",LeftRight,RightTicks("%#.1f"));


