import graph3;

currentprojection=orthographic(5,4,2);

size(0,150);
patch s=octant1x;
draw(surface(s),green+opacity(0.5));
draw(s.external(),blue);

triple[][] P=s.P;

for(int i=0; i < 4; ++i)
  dot(P[i],red);

axes3("$x$","$y$",Label("$z$",align=Z));
triple P00=P[0][0];
triple P10=P[1][0];
triple P01=P[0][1];
triple P02=P[0][2];
triple P11=P[1][1];
triple P12=P[1][2];
triple Q11=XYplane(xypart(P11));
triple Q12=XYplane(xypart(P12));

draw(P11--Q11,dashed);
draw(P12--Q12,dashed);
draw(O--Q12--Q11--(Q11.x,0,0));
draw(Q12--(Q12.x,0,0));

label("$(1,0,0)$",P00,-2Y);
label("$(1,a,0)$",P10,-Z);
label("$(1,0,a)$",P01,-2Y);
label("$(a,0,1)$",P02,Z+X-Y);
label("$(1,a,a)$",P11,3X);
label("$(a,a^2,1)$",P12,7X+Y);
