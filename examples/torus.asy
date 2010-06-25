size(200);
import graph3;

currentprojection=perspective(5,4,4);

real R=3;
real a=1;

/*
import solids;
revolution torus=revolution(reverse(Circle(R*X,a,Y,32)),Z,90,345);
surface s=surface(torus);
*/

triple f(pair t) {
  return ((R+a*cos(t.y))*cos(t.x),(R+a*cos(t.y))*sin(t.x),a*sin(t.y));
}

surface s=surface(f,(radians(90),0),(radians(345),2pi),8,8,Spline);
draw(s,green,render(compression=Low,merge=true));
