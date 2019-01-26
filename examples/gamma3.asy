import graph3;
import palette;

size(12cm,IgnoreAspect);
currentprojection=orthographic(1,-2,1);

real X=4.5;
real M=abs(gamma((X,0)));

pair Gamma(pair z) 
{
  return (z.x > 0 || z != floor(z.x)) ? gamma(z) : M;
}

real f(pair z) {return min(abs(Gamma(z)),M);}

surface s=surface(f,(-2.1,-2),(X,2),70,Spline);

real Arg(triple v)
{
  return degrees(Gamma((v.x,v.y)),warn=false);
}

s.colors(palette(s.map(Arg),Wheel()));
draw(s,render(compression=Low,merge=true));

real xmin=point((-1,-1,-1)).x;
real xmax=point((1,1,1)).x;
draw((xmin,0,0)--(xmax,0,0),dashed);

xaxis3("$\mathop{\rm Re} z$",Bounds,InTicks);
yaxis3("$\mathop{\rm Im} z$",Bounds,InTicks(beginlabel=false));
zaxis3("$|\Gamma(z)|$",Bounds,InTicks);
