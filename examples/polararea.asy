import math;
import graph;

size(0,150);

real f(real t) {return 5+cos(10*t);}

xaxis("$x$");
yaxis("$y$");

real theta1=pi/8;
real theta2=pi/3;
path k=graph(f,theta1,theta2,operator ..);
real rmin=min(k).y;
real rmax=max(k).y;
draw((0,0)--rmax*expi(theta1),dotted);
draw((0,0)--rmax*expi(theta2),dotted);

path g=polargraph(f,theta1,theta2,operator ..);
path h=(0,0)--g--cycle;
fill(h,lightgray);
draw(h);

real thetamin=3*pi/10;
real thetamax=2*pi/10;
pair zmin=polar(f(thetamin),thetamin);
pair zmax=polar(f(thetamax),thetamax);
draw((0,0)--zmin,dotted+red);
draw((0,0)--zmax,dotted+blue);

draw("$\theta_*$",arc((0,0),0.5*rmin,0,degrees(thetamin)),red+fontsize(10pt),
     PenMargins);
draw("$\theta^*$",arc((0,0),0.5*rmax,0,degrees(thetamax)),blue+fontsize(10pt),
     PenMargins);

draw(arc((0,0),rmin,degrees(theta1),degrees(theta2)),red,PenMargins);
draw(arc((0,0),rmax,degrees(theta1),degrees(theta2)),blue,PenMargins);

