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

labeldot("$O$",O,S,red);
labeldot("$A$",A,dir(C--A,B--A),red);
labeldot("$B$",B,dir(C--B,A--B),red);
labeldot("$C$",C,dir(A--C,B--C),red);
labeldot("$D$",D,red);

shipout("problem6a");

erase();
size(7cm,0);

real theta=degrees(asin(0.5/sqrt(7)));

B=(0,r2);
A=B+2sqrt(3)*dir(270-theta);
C=A+sqrt(21);

pair Ap=extension(A,O,B,C);
pair Bp=extension(B,O,C,A);
pair Cp=extension(C,O,A,B);

draw(A--B--C--cycle);

currentpen=black;

draw("1",A--O,-0.25*I*dir(A--O));
draw(O--Ap);
draw("$\sqrt{7}$",B--O,LeftSide);
draw(O--Bp);
draw("4",C--O);
draw(O--Cp);

labeldot("$O$",O,1.5*dir(B--Bp,Cp--C),red);
labeldot("$A$",A,1.5*dir(C--A,B--A),red);
labeldot("$B$",B,NW,red);
labeldot("$C$",C,dir(A--C,B--C),red);
labeldot("$A'$",Ap,dir(A--Ap),red);
labeldot("$B'$",Bp,dir(B--Bp),red);
labeldot("$C'$",Cp,dir(C--Cp),red);

perpendicular(Ap,Ap--O,blue);
perpendicular(Bp,Bp--C,blue);
perpendicular(Cp,Cp--O,blue);

shipout();
