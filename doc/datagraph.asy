import graph;

size(400,200,IgnoreAspect);

pair[] f={(5,5),(50,20),(90,90)};
pair[] df={(0,0),(5,7),(0,5)};

pair[] f2={(10,40),(20,80),(40,30),(60,60)};

frame dot;
filldraw(dot,scale(0.8mm)*unitcircle,blue);

frame mark;
filldraw(mark,scale(0.8mm)*polygon(6),green);
draw(mark,scale(0.8mm)*cross(6),blue);

errorbars(f,df,red);
draw(graph(f),dot,false);
draw(shift(0,10)*graph(f2),mark);

xaxis("$x$",BottomTop,LeftTicks);
yaxis("$y$",LeftRight,RightTicks);

xaxis(Dotted,YEquals(60.0,false));
yaxis(Dotted,XEquals(80.0,false));

