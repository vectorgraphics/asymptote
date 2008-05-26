real euler(real y, real f(real x, real y), real a, real b=a, int n=0,
	   real h=0, bool dynamic=false, real tolmin=0, real tolmax=0)
{
  if(h == 0) {
    if(b == a) return y;
    if(n == 0) abort("Either n or h must be specified");
    else h=(b-a)/n;
  }
  real x=a;
  for(int i=0; i < n; ++i) {
    y += h*f(x,y);
    x += h;
  }
  return y;
}
