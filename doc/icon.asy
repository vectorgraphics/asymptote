import graph;

size(30,30,IgnoreAspect);

real f(real t) {return t < 0 ? -1/t : -0.5/t;}

picture logo(pair s=0, pen q) 
{
  picture pic;
  pen p=linewidth(3)+q;
  real a=-0.5;
  real b=1;
  real eps=0.1;
  draw(pic,shift((eps,-f(a)))*graph(f,a,-eps),p);
  real c=0.5*a;
  pair z=(0,f(c)-f(a));
  draw(pic,z+c+eps--z,p);
  yaxis(pic,p);
  return shift(s)*pic;
} 

add(logo(red));
