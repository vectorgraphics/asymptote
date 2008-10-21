import graph3;

size(200,0);
currentprojection=perspective(10,-50,5);

// From http://local.wasp.uwa.edu.au/~pbourke/surfaces_curves/klein/
triple f(pair t) {
  real u=t.x;
  real v=t.y;
  real r=4*(1-cos(u)/2);
  real x=6*cos(u)*(1+sin(u)) + (u < pi ? r*cos(u)*cos(v) : r*cos(v+pi));
  real y=16*sin(u) + (u < pi ? r*sin(u)*cos(v) : 0);
  real z=r*sin(v);
  return (x,y,z);
}

pen p=rgb(0.2,0.5,0.7);

draw(surface(f,(0,0),(2pi,2pi),60,60),lightgray,meshpen=p);
