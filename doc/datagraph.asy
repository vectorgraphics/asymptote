import graph;

size(400,200,IgnoreAspect);

pair[] f={(1,1),(50,20),(90,90)};
pair[] df={(0,0),(5,10),(0,5)};

guide g=graph(f);
errorbars(f,df,red);
draw(g);
dot(g);

xaxis("$x$",BottomTop,LeftTicks);
yaxis("$y$",LeftRight,RightTicks);

xaxis(Dotted,YEquals(60.0,false));
yaxis(Dotted,XEquals(80.0,false));


