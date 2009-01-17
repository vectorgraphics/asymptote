import three;
import math;
texpreamble("\usepackage{bm}");

size(300,0);

pen thickp=linewidth(0.5mm);
real radius=0.8, lambda=37, aux=60;

currentprojection=perspective(4,1,2); 

// Planes
pen bg=gray(0.9)+opacity(0.5);
draw(surface((1.2,0,0)--(1.2,0,1.2)--(0,0,1.2)--(0,0,0)--cycle),bg);
draw(surface((0,1.2,0)--(0,1.2,1.2)--(0,0,1.2)--(0,0,0)--cycle),bg);
draw(surface((1.2,0,0)--(1.2,1.2,0)--(0,1.2,0)--(0,0,0)--cycle),bg);

real r=1.5;
pen p=rgb(0,0.7,0);
draw(Label("$x$",1),O--r*X,p,Arrow3);
draw(Label("$y$",1),O--r*Y,p,Arrow3);
draw(Label("$z$",1),O--r*Z,p,Arrow3);
label("$\rm O$",(0,0,0),W);
  
// Point Q
triple pQ=radius*dir(lambda,aux);
draw(O--radius*dir(90,aux),dashed);
label("$\rm Q$",pQ,N+3*W);
draw("$\lambda$",arc(O,0.15pQ,0.15*Z),N+0.3E);

// Particle
triple m=pQ-(0.26,-0.4,0.28);
real width=5;
dot("$m$",m,SE,linewidth(width));
draw("$\bm{\rho}$",(0,0,0)--m,Arrow3,PenMargin3(0,width));
draw("$\bm{r}$",pQ--m,Arrow3,PenMargin3(0,width));

// Spherical octant
real r=sqrt(pQ.x^2+pQ.y^2);
draw(arc((0,0,pQ.z),(r,0,pQ.z),(0,r,pQ.z)),dashed);
draw(arc(O,radius*Z,radius*dir(90,aux)),dashed);
draw(arc(O,radius*Z,radius*X),thickp);
draw(arc(O,radius*Z,radius*Y),thickp);
draw(arc(O,radius*X,radius*Y),thickp);

// Moving axes
triple i=dir(90+lambda,aux);
triple k=unit(pQ);
triple j=cross(k,i);

draw(Label("$x$",1),pQ--pQ+0.2*i,2W,red,Arrow3);
draw(Label("$y$",1),pQ--pQ+0.32*j,red,Arrow3);
draw(Label("$z$",1),pQ--pQ+0.26*k,red,Arrow3);

draw("$\bm{R}$",O--pQ,Arrow3,PenMargin3);
draw("$\omega\bm{K}$",arc(0.9Z,0.2,90,-120,90,160,CW),1.2N,Arrow3);
