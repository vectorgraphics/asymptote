import markers;
import geometry;
import math;

size(10cm,0);
pair A=0, B=(1,0), C=(.7,1), D=(-.5,0), F=rotate(-90)*(C-B)/2+B;

draw(A--B);
draw(A--C);
pen p=1mm+black;
draw(B--C,p);
draw(A--D);
draw(B--F,p);
label("$A$",A,SW);
label("$B$",B,SE);
label("$C$",C,N);
dot(Label("$D$",D,N+NE));
dot(Label("$F$",F,N+NW));

markangle(C,A,B);

markangle(scale(1.5)*"$\theta$", radius=40, B,A,C,
          ArcArrow(2mm), p=1mm+red);

markangle(Label("$\gamma$",Relative(.25)),n=2, radius=-30,
          C,A,B, p=.7blue+2);

markangle(n=3, A,B,C, markersuniform(markinterval=stickframe(n=2)));

pen RedPen=.7red+1bp;
markangle(A,D,C, p=RedPen, markersuniform(3,markinterval=stickframe(3,4mm,RedPen)));
drawline(A,A+dir(A--D,A--C),dotted);

perpendicular(B,NE,F-B,size=10mm, p=1mm+red, TrueMargin(linewidth(p)/2,linewidth(p)/2),Fill(yellow));
