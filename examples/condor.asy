// Peter Luschny's Condor function
// http://www.luschny.de/math/asy/ElCondorYElGamma.html

import palette;
import graph3;

size(300,300,IgnoreAspect);
currentprojection=orthographic(0,-1,0,center=true);
currentlight=White;
real K=7;

triple condor(pair t)
{
  real y=t.y;
  real x=t.x*y;
  real e=gamma(y+1);
  real ymx=y-x;
  real ypx=y+x;
  real a=gamma((ymx+1)/2);
  real b=gamma((ymx+2)/2);
  real c=gamma((ypx+1)/2);
  real d=gamma((ypx+2)/2);
  real A=cos(pi*ymx);
  real B=cos(pi*ypx);
  return (x,y,log(e)+log(a)*((A-1)/2)+log(b)*((-A-1)/2)+log(c)*((B-1)/2)+
          log(d)*((-B-1)/2));
}

surface s=surface(condor,(-1,0),(1,K),16,Spline);
s.colors(palette(s.map(zpart),Rainbow()));

draw(s,render(compression=Low,merge=true));
