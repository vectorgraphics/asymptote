import graph;

size(400,200,IgnoreAspect);

pair f[]={(1,1),(50,20),(100,100)};

draw(graph(f));

xaxis("$x$",BottomTop,LeftTicks);
yaxis("$y$",LeftRight,RightTicks);

shipout();

