import graph3;
size(200,0);

currentlight=Viewport;

triple f(pair t) {
  real u=log(abs(tan(t.y/2)));
  return (10*sin(t.y),cos(t.x)*(cos(t.y)+u),sin(t.x)*(cos(t.y)+u));
}

surface s=surface(f,(0,pi/2),(2pi,pi-0.1),7,15,Spline);
draw(s,olive+0.25*white,render(compression=Low,merge=true));
