import math;
size(0,100);
real theta=30;

pair A=(0,0); 
pair B=dir(theta);
pair C=(1,0);
pair D=(1,Tan(theta));
pair E=(Cos(theta),0);

filldraw(A--C{N}..B--cycle,lightgrey);
draw(B--C--D--cycle);
draw(B--E);

draw("$x$",arc(C,A,B,0.7),RightSide,Arrow,PenMargin);

labeldot("$A$",A,W);
labeldot("$B$",B,NW);
labeldot("$C$",C);
labeldot("$D$",D);
labeldot(("$E$"),E,S);
label("$1$",A--B,LeftSide);
      
shipout();
