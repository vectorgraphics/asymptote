import graph;

size(200,150,IgnoreAspect);

file in=line(input("filegraph.dat"));
real[][] a=dimension(in,0,0);
a=transpose(a);

real[] x=a[0];
real[] y=a[1];

draw(graph(x,y),red);

xaxis("$x$",BottomTop,LeftTicks);
yaxis("$y$",LeftRight,RightTicks);
