import graph;

picture pic=new picture;
real xsize=200, ysize=140;
size(pic,xsize,ysize,IgnoreAspect);

pair[] f={(5,5),(50,20),(90,90)};
pair[] df={(0,0),(5,7),(0,5)};

errorbars(pic,f,df,red);
draw(pic,graph(pic,f),marker(scale(0.8mm)*unitcircle,blue,Fill),Below);

xaxis(pic,"$x$",BottomTop,LeftTicks);
yaxis(pic,"$y$",LeftRight,RightTicks);

picture pic2=new picture;
size(pic2,xsize,ysize,IgnoreAspect);

frame mark;
filldraw(mark,scale(0.8mm)*polygon(6),green);
draw(mark,scale(0.8mm)*cross(6),blue);

draw(pic2,graph(f),mark);

xaxis(pic2,"$x$",BottomTop,LeftTicks);
yaxis(pic2,"$y$",LeftRight,RightTicks);

xaxis(pic2,red+Dotted,YEquals(55.0,false));
yaxis(pic2,red+Dotted,XEquals(70.0,false));

// Fit pic to W of origin:
add(pic.fit(W)); 

// Fit pic2 to E of (5mm,0):
add((5mm,0),pic2.fit(E));

