size(0,150);
import math;

real a=3;
real b=4;
real c=hypot(a,b);

pair z1=(0,b);
pair z2=(a,0);
pair z3=(a+b,0);
draw(square((0,0),z3));
draw(square(z1,z2));
perpendicular(z1,z1--z2,blue);
perpendicular(z3,N,blue);

real d=0.3;
pair v=unit(z2-z1);
draw(baseline("$a$"),-d*I--z2-d*I,red,Bars,Arrows);
draw(baseline("$b$"),z2-d*I--z3-d*I,red,Arrows,Bars);
draw("$c$",z3+z2*I-d*v--z2-d*v,red,Arrows);
draw("$a$",z3+d--z3+z2*I+d,red,Arrows,Bars);
draw("$b$",z3+z2*I+d--z3+z3*I+d,red,Arrows,Bars);

shipout();

