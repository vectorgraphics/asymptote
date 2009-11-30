import graph;
import animation;

size(0,200);

defaultpen(3);
dotfactor=4;

pair wheelpoint(real t)
{
  return (t+cos(t),-sin(t));
}

guide wheel(guide g=nullpath, real a, real b, int n)
{
  real width=(b-a)/n;
  for(int i=0; i <= n; ++i) {
    real t=a+width*i;
    g=g--wheelpoint(t);
  }
  return g;
}

real t1=0; 
real t2=t1+2*pi;

draw(circle((0,0),1));
draw(wheel(t1,t2,100),linetype("0 2"));
yequals(Label("$y=-1$",1.0),-1,extend=true,linetype("4 4"));
xaxis(Label("$x$",align=3SW),0);
yaxis("$y$",0,1.2);
pair z1=wheelpoint(t1);
pair z2=wheelpoint(t2);
dot(z1);
dot(z2);

animation a;

int n=10;
real dt=(t2-t1)/n;
for(int i=0; i <= n; ++i) {
  save();
  
  real t=t1+dt*i;
  draw(circle((t,0),1),red);
  dot(wheelpoint(t));

  a.add(); // Add currentpicture to animation.
  restore();
}

// Produce the final merged gif.
a.movie(BBox(0.25cm),loops=10,delay=250);
