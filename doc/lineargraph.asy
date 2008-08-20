import graph;

size(250,200,IgnoreAspect);

real Sin(real t) {return sin(2pi*t);}
real Cos(real t) {return cos(2pi*t);}

draw(graph(Sin,0,1),red,"$\sin(2\pi x)$");
draw(graph(Cos,0,1),blue,"$\cos(2\pi x)$");

xaxis("$x$",BottomTop,LeftTicks);
yaxis("$y$",Right,LeftTicks(trailingzero));
yaxis("",Left);

label("LABEL",point(0),UnFill(1mm));

picture pic;
size(pic,250,200,IgnoreAspect);
add(pic,currentpicture.fit(),(0,0),W);
add(pic,legend(currentpicture),(0,0),20E,UnFill);
shipout(pic);
