import graph3;
import palette;

size(12cm,IgnoreAspect);
currentprojection=orthographic(1,-2,1);

real f(pair z) {return abs(sin(z));}

real Arg(triple v) {return degrees(cos((v.x,v.y)),warn=false);}

surface s=surface(f,(-pi,-2),(pi,2),20,Spline);

s.colors(palette(s.map(Arg),Wheel()));
draw(s,render(compression=Low,merge=true));

real xmin=point((-1,-1,-1)).x;
real xmax=point((1,1,1)).x;
draw((xmin,0,0)--(xmax,0,0),dashed);

xaxis3("$\mathop{\rm Re} z$",Bounds,InTicks);
yaxis3("$\mathop{\rm Im} z$",Bounds,InTicks(beginlabel=false));
zaxis3("$|\sin(z)|$",Bounds,InTicks);
