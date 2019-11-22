import geometry;
import math;

size(7cm,0);

if(!settings.xasy && settings.outformat != "svg") settings.tex="pdflatex";

real theta=degrees(asin(0.5/sqrt(7)));

pair B=(0,sqrt(7));
pair A=B+2sqrt(3)*dir(270-theta);
pair C=A+sqrt(21);
pair O=0;

pair Ap=extension(A,O,B,C);
pair Bp=extension(B,O,C,A);
pair Cp=extension(C,O,A,B);

perpendicular(Ap,NE,Ap--O,blue);
perpendicular(Bp,NE,Bp--C,blue);
perpendicular(Cp,NE,Cp--O,blue);

draw(A--B--C--cycle);

draw("1",A--O,-0.25*I*dir(A--O));
draw(O--Ap);
draw("$\sqrt{7}$",B--O,LeftSide);
draw(O--Bp);
draw("4",C--O);
draw(O--Cp);

dot("$O$",O,dir(B--Bp,Cp--C),red);
dot("$A$",A,dir(C--A,B--A),red);
dot("$B$",B,NW,red);
dot("$C$",C,dir(A--C,B--C),red);
dot("$A'$",Ap,dir(A--Ap),red);
dot("$B'$",Bp,dir(B--Bp),red);
dot("$C'$",Cp,dir(C--Cp),red);

label(graphic("piicon.png","width=2.5cm, bb=0 0 147 144"),Ap,5ENE);
