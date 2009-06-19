size(200);
import graph3;

currentprojection=perspective(5,4,4);

//import solids;
//revolution torus=revolution(shift(3X)*Circle(O,1,Y,32),Z,90,345);
//draw(surface(torus),green);

real R=3;
real a=1;

triple f(pair t) {
  return ((R+a*cos(t.y))*cos(t.x),(R+a*cos(t.y))*sin(t.x),a*sin(t.y));
}

draw(surface(f,(radians(90),0),(radians(345),2pi),8,8,Spline),green);
