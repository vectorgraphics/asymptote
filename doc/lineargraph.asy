
import graph;

picture pic=new picture;

size(pic,250,200,IgnoreAspect);

real Sin(real t) {return sin(2pi*t);}
real Cos(real t) {return cos(2pi*t);}

draw(pic,graph(pic,Sin,0,1),red,"$\sin(2\pi x)$");
draw(pic,graph(pic,Cos,0,1),blue,"$\cos(2\pi x)$");

xaxis(pic,"$x$",BottomTop,LeftTicks);
yaxis(pic,"$y$",LeftRight,RightTicks);

add(pic.fit(W));
add(point(E),legend(pic,20E));
