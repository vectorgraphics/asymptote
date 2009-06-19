import graph3;
size(200,0);

triple f(pair t) {
  return(10*sin(t.y),cos(t.x)*(cos(t.y)+log(abs(tan(t.y/2)))),
         sin(t.x)*(cos(t.y)+log(abs(tan(t.y/2)))));
}

surface s=surface(f,(0,pi/2),(2pi,pi-0.1),7,15,Spline);
draw(s,olive);
