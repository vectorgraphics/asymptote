import palette;
import graph3;

settings.prc = false;
size(300, 300, IgnoreAspect);
currentlight = adobe;
real K = 6;

bool cond(pair z)
{
    return (abs(z.x) < z.y)
    && (-K <= z.x) && (z.x <= K)
    && ( 0 <= z.y) && (z.y <= K);
}

real StealthDragon(pair p)
{
    if(abs(p.x) == p.y) return 0;
    real e = gamma(p.y + 1);
    real ymx = p.y - p.x;
    real ypx = p.y + p.x;
    real a = gamma((ymx + 1)/2);
    real b = gamma((ymx + 2)/2);
    real c = gamma((ypx + 1)/2);
    real d = gamma((ypx + 2)/2);
    real A = cos(pi*ymx);
    real B = cos(pi*ypx);
    return log(e)+log(a)*((A-1)/2)+log(b)*((-A-1)/2)
           +log(c)*((B-1)/2)+log(d)*((-B-1)/2);
}

surface s = surface(StealthDragon, (-K,0), (K,K), 96, cond);
draw(s,mean(palette(s.map(zpart), Rainbow())), black);
