import graph3;
import palette;

size(12cm,IgnoreAspect);
currentprojection=orthographic(1,-2,1);

real X=4.5;
real M=abs(gamma((X,0)));

pair gamma0(pair z) 
{
  return z.x <= 0 && z == floor(z.x) ? M : gamma(z);
}

surface s=surface(new real(pair z) {return min(abs(gamma0(z)),M);},
		  (-2.1,-2),(X,2),70,Spline);

s.colors(palette(s.map(new real(triple v) {
	return degrees(gamma0((v.x,v.y)),warn=false);}),Wheel()));
draw(s);

xaxis3("$\mathop{\rm Re} z$",Bounds,InTicks(Label));
yaxis3("$\mathop{\rm Im} z$",Bounds,InTicks(beginlabel=false,Label));
zaxis3("$|\Gamma(z)|$",Bounds,InTicks());
