import graph;

size(200,150,IgnoreAspect);

// Break the axis at 3; restart at 8.
real a=3, b=8;

scale(Broken(a,b),Linear);

real[] x={1,2,10};
real[] y=x^2;

draw(graph(x,y),red,MarkFill[0]);

xaxis("$x$",BottomTop,LeftTicks(new real[]{0,1,2,9,10}));
yaxis("$y$",LeftRight,RightTicks);

label(rotate(90)*Break,(a,point(S).y));
label(rotate(90)*Break,(a,point(N).y));

