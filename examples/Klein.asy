import graph3;

size(200,0);
currentprojection=perspective(20,-50,20);

// From http://local.wasp.uwa.edu.au/~pbourke/geometry/klein/
triple f(pair t) {
  real u=t.x;
  real v=t.y;
  real r=2-cos(u);
  real x=3*cos(u)*(1+sin(u))+r*cos(v)*(u < pi ? cos(u) : -1);
  real y=8*sin(u)+(u < pi ? r*sin(u)*cos(v) : 0);
  real z=r*sin(v);
  return (x,y,z);
}

draw(surface(f,(0,0),(2pi,2pi),8,8,Spline),lightgray);
