import graph3;
import palette;
size(200);
currentprojection=orthographic(6,8,2);
viewportmargin=(1cm,0);
 
real c0=0.1;

real f(real r) {return r*(1-r/6)*exp(-r/3);}

triple f(pair t) {
  real r=t.x;
  real phi=t.y;
  real f=f(r);
  real s=max(min(f != 0 ? c0/f : 1,1),-1);
  real R=r*sqrt(1-s^2);
  return (R*cos(phi),R*sin(phi),r*s);
}

bool cond(pair t) {return f(t.x) != 0;}

real R=abs((20,20,20));
surface s=surface(f,(0,0),(R,2pi),100,8,Spline,cond);

s.colors(palette(s.map(abs),Gradient(palegreen,heavyblue)));

render render=render(compression=Low,merge=true);
draw(s,render);
draw(zscale3(-1)*s);
 
axes3("$x$","$y$","$z$",Arrow3);
