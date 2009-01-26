import graph;
size(300,IgnoreAspect);

bool3 branch(real x)
{
  static int lastsign=0;
  if(x <= 0 && x == floor(x)) return false;
  int sign=sgn(gamma(x));
  bool b=lastsign == 0 || sign == lastsign;
  lastsign=sign;
  return b ? true : default;
}

draw(graph(gamma,-4,4,n=2000,branch),red);
 
scale(false);
xlimits(-4,4);
ylimits(-6,6);
crop();

xaxis("$x$",RightTicks(NoZero));
yaxis(LeftTicks(NoZero));

label("$\Gamma(x)$",(1,2),red);
