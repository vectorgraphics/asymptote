size(7cm,0);
import math;

real theta=degrees(asin(0.5/sqrt(7)));

pair B=(0,sqrt(7));
pair A=B+2sqrt(3)*dir(270-theta);
pair C=A+sqrt(21);
pair O=0;

pair Ap=extension(A,O,B,C);
pair Bp=extension(B,O,C,A);
pair Cp=extension(C,O,A,B);

perpendicular(Ap,Ap--O,blue);
perpendicular(Bp,Bp--C,blue);
perpendicular(Cp,Cp--O,blue);

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

label(include("sflogo.eps","width=4cm"),Ap,6*NE,red);

