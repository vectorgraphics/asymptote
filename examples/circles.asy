size(6cm,0);
import math;

currentpen=magenta;

real r1=1;
real r2=sqrt(7);
real r3=4;
pair O=0;

path c1=circle(O,r1);
draw(c1,green);
draw(circle(O,r2),green);
draw(circle(O,r3),green);

real x=-0.6;
real y=-0.8;
real yD=0.3;
pair A=(sqrt(r1^2-y^2),y);
pair B=(-sqrt(r2^2-y^2),y);
pair C=(x,sqrt(r3^2-x^2));

pair d=A+r2*dir(B--C);
pair D=intersectionpoint(c1,A--d);

draw(A--B--C--cycle);
draw(interp(A,D,-0.5)--interp(A,D,1.5),blue);

dot("$O$",O,S,red);
dot("$A$",A,dir(C--A,B--A),red);
dot("$B$",B,dir(C--B,A--B),red);
dot("$C$",C,dir(A--C,B--C),red);
dot("$D$",D,red);
