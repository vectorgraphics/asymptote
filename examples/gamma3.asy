import graph3;
import palette;

size(12cm,IgnoreAspect);
currentprojection=orthographic(1,-2,1);

real[] p={0.99999999999980993,676.5203681218851,-1259.1392167224028,771.32342877765313,-176.61502916214059,12.507343278686905,-0.13857109526572012,9.9843695780195716e-6,1.5056327351493116e-7};

pair gamma(pair z) {
  if(z.x < 0.5)
    return pi/(sin(pi*z)*gamma(1.0-z));
  z -= 1.0;
  pair x=p[0];
  for(int i=1; i < p.length; ++i)
    x += p[i]/(z+(i,0));
  pair t=p.length-1.5+z;
  return sqrt(2*pi)*t^(z+0.5)*exp(-t)*x;
}

real X=4.5;
real M=abs(gamma((X,0)));

pair gamma0(pair z) 
{
  return z.x <= 0 && z == floor(z.x) ? M : gamma(z);
}

surface s=surface(new real(pair z) {return min(abs(gamma0(z)),M);},
		  (-2.1,-2),(X,2),70,Spline);
s.colors(palette(s.map(new real(triple v) {return angle(gamma0((v.x,v.y)));}),
		 Wheel()));
draw(s);

xaxis3("$x$",Bounds,InTicks(Label));
yaxis3("$y$",Bounds,InTicks(beginlabel=false,Label));
zaxis3("$z$",Bounds,InTicks());
