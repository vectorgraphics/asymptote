// Peter Luschny's Condor function
// Reference: http: //www.luschny.de/math/asy/ElCondorYElGamma.html

import palette;
import graph3;

size(300,300,IgnoreAspect);
currentlight=adobe;
real K=7;

bool cond(pair z)
{
  return(abs(z.x) < z.y)
    && (-K <= z.x) && (z.x <= K)
    && (0 <= z.y) && (z.y <= K);
}

real condor(pair p)
{
  real e=gamma(p.y+1);
  real ymx=p.y-p.x;
  real ypx=p.y+p.x;
  real a=gamma((ymx+1)/2);
  real b=gamma((ymx+2)/2);
  real c=gamma((ypx+1)/2);
  real d=gamma((ypx+2)/2);
  real A=cos(pi*ymx);
  real B=cos(pi*ypx);
  return log(e)+log(a)*((A-1)/2)+log(b)*((-A-1)/2)+log(c)*((B-1)/2)+
    log(d)*((-B-1)/2);
}

surface s=surface(condor,(-K,0),(K,K),96,cond);
s.colors(palette(s.map(zpart),Rainbow()));

draw(s,meshpen=black);
