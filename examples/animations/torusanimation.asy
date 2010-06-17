import graph3;
import animation;
import solids;

currentprojection=perspective(50,40,20);

currentlight=(0,5,5);

real R=3;
real a=1;
int n=8;

path3[] p=new path3[n];
animation A;
 
for(int i=0; i < n; ++i) {
  triple g(real s) {
    real twopi=2*pi;
    real u=twopi*s;
    real v=twopi/(1+i+s);
    real cosu=cos(u);
    return((R-a*cosu)*cos(v),(R-a*cosu)*sin(v),-a*sin(u));
  } 
  p[i]=graph(g,0,1,operator ..);
}

triple f(pair t) {
  real costy=cos(t.y);
  return((R+a*costy)*cos(t.x),(R+a*costy)*sin(t.x),a*sin(t.y));
}

surface s=surface(f,(0,0),(2pi,2pi),8,8,Spline);
 
for(int i=0; i < n; ++i){
  picture fig;
  size(fig,20cm);
  draw(fig,s,yellow);
  for(int j=0; j <= i; ++j)
    draw(fig,p[j],blue+linewidth(4));
  A.add(fig);
}

A.movie(BBox(10,Fill(rgb(0.98,0.98,0.9))),delay=100);
