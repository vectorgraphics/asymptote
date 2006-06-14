import graph;
import math;

size(140,80,IgnoreAspect);

picture logo(pair s=0, pen q) 
{
  picture pic;
  pen p=linewidth(2)+fontsize(24)+q;
  real a=-0.4;
  real b=0.95;
  real eps=0.1;
  real y=5;
  path A=(a,0){dir(10)}::{dir(89.5)}(0,3y/2);
  draw(pic,A,p);
  draw(pic,(0,-y){dir(88.3)}::{dir(20)}(b,0),p);
  real c=0.5*a;
  pair z=(0,2.5);
  label(pic,"{\it symptote}",z,0.25*E+0.83S,p);
  pair w=(0,1.7);
  draw(pic,intersectionpoint(A,w-1--w)--w,p);
  axes(pic,p);
  return shift(s)*pic;
} 

pair z=(-0.015,0.08);
for(real x=0; x < 1; x += 0.1) {
  add(logo(x*z,gray(0.4*x)));
} 
add(logo(red));
