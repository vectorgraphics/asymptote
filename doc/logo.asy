import graph;

size(150,80,IgnoreAspect);

real f(real t) {return t < 0 ? -1/t : -0.5/t;}

picture logo(pair s=0, pen q) 
{
  picture pic=new picture;
  pen p=linewidth(2)+fontsize(24)+q;
  real a=-0.5;
  real b=1;
  real eps=0.1;
  draw(pic,shift((eps,-f(a)))*graph(f,a,-eps),p);
  draw(pic,shift(-(eps,f(b)))*graph(f,eps,b),p);
  real c=0.5*a;
  pair z=(0,f(c)-f(a));
  label(pic,"{\it symptote}",z,0.25*E+0.5*S,p);
  draw(pic,z+c+eps--z,p);
  axes(pic,p);
  return shift(s)*pic;
} 

pair z=(-0.015,0.08);
for(real x=0; x < 1; x += 0.1) {
  add(logo(x*z,gray(0.4*x)));
} 
add(logo(red));

shipout();
