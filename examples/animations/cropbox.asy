// Carry a crop box once around a figure-eight Klein bottle. The surface is
// drawn twice: ghosted whole, and cropped to the box, so that the patches
// caught by the box can be watched appearing, being cut and disappearing as
// it passes. The box is axis aligned, so cropping to it is six plane crops
// in turn; where two of its faces meet, the corner they cut is the meeting
// of two of those crops.
import graph3;
import crop3;
import animation;

size(10cm);
currentprojection=orthographic((2.6,1.7,1.4));
currentlight=Headlamp;

real R=2;

triple klein(pair z)
{
  real u=z.x, v=z.y;
  real r=R+cos(u/2)*sin(v)-sin(u/2)*sin(2v);
  return (r*cos(u),r*sin(u),sin(u/2)*sin(v)+cos(u/2)*sin(2v));
}

surface bottle=surface(klein,(0,0),(2pi,2pi),24,12,Spline);

animation a;

int n=48;
triple halfsize=(0.75,0.9,0.85);

for(int i=0; i < n; ++i) {
  real phi=2pi*i/n;
  // Follow the tube of the bottle, rising and falling as it goes.
  triple centre=(R*cos(phi),R*sin(phi),0.7*sin(3phi));

  picture pic;
  size(pic,10cm);
  draw(pic,bottle,gray(0.5)+opacity(0.07));
  draw(pic,crop(bottle,boxsides(centre-halfsize,centre+halfsize)),
       orange+opacity(0.95),meshpen=black+0.4pt);
  draw(pic,box(centre-halfsize,centre+halfsize),red+0.6pt);
  a.add(pic);
}

a.movie(loops=0,delay=320);
