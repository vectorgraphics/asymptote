import graph;

picture pic=new picture;
real xsize=200, ysize=140;
size(pic,xsize,ysize,IgnoreAspect);

pair[] f={(5,5),(50,20),(90,90)};
pair[] df={(0,0),(5,7),(0,5)};

frame dot;
filldraw(dot,scale(0.8mm)*unitcircle,blue);

errorbars(pic,f,df,red);
draw(pic,graph(pic,f),dot,false);

xaxis(pic,"$x$",BottomTop,LeftTicks);
yaxis(pic,"$y$",LeftRight,RightTicks);

xaxis(pic,Dotted,YEquals(60.0,false));
yaxis(pic,Dotted,XEquals(80.0,false));



picture pic2=new picture;
size(pic2,xsize,ysize,IgnoreAspect);

frame mark;
filldraw(mark,scale(0.8mm)*polygon(6),green);
draw(mark,scale(0.8mm)*cross(6),blue);

draw(pic2,graph(f),mark);

xaxis(pic2,"$x$",BottomTop,LeftTicks);
yaxis(pic2,"$y$",LeftRight,RightTicks);

// Fit pic to W of origin:
add(pic.fit(W)); 

// Fit pic2 to E of (5mm,0):
add((5mm,0),pic2.fit(E));

