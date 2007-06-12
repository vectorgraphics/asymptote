import graph;

size(140,80,IgnoreAspect);

picture logo(pair s=0, pen q) 
{
  picture pic;
  pen p=linewidth(2)+fontsize(24)+q;
  real a=-0.4;
  real b=0.95;
  real y=5;
  path A=(a,0){dir(10)}::{dir(89.5)}(0,3y/2);
  draw(pic,A,p);
  draw(pic,(0,-y){dir(88.3)}::{dir(20)}(b,0),p);
  real c=0.5*a;
  pair z=(0,2.5);
  label(pic,"{\it symptote}",z,0.25*E+0.169S,p);
  pair w=(0,1.7);
  draw(pic,intersectionpoint(A,w-1--w)--w,p);
  axes(pic,p);
  return shift(s)*pic;
} 

pair z=(-0.015,0.08);
for(int x=0; x < 10; ++x)
  add(logo(0.1*x*z,gray(0.04*x)));

add(logo(red));
