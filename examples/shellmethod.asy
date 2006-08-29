import graph3;
import solids;
size(0,150);
currentprojection=perspective(0,0,10);
currentlight=(3,10,0);
pen color=green;
real alpha=240;

real f(real x) {return 2x^2-x^3;}
triple F(real x) {return (x,f(x),0);}

int n=20;
path3[] blocks=new path3[n];
for(int i=1; i <= n; ++i) {
  real height=f((i-0.5)*2/n);
  real left=(i-1)*2/n;
  real right=i*2/n;
  blocks[i-1]=
    (left,0,0)--(left,height,0)--(right,height,0)--(right,0,0)--cycle3;
}

guide3 p=graph(F,0,2,n=30)--cycle3;

revolution a=revolution(p,Y,0,alpha);
a.filldraw(color,1,blue,false);
filldraw(p,color);
filldraw(rotate(alpha,(0,1,0))*p,color);
for(int i=0; i < n; ++i) {
  filldraw(blocks[i],white);
  draw(blocks[i]);
}
draw(p);

bbox3 b=autolimits(O,2.2X+1.5Y+Z);

xaxis(Label("$x$",1),b,Arrow);
yaxis(Label("$y$",1),b,dashed,Arrow);
arrow("$y=2x^2-x^3$",F(1.8),NE,0.75cm);
draw(arc(1.22Y,0.3,90,0,7.5,180),ArcArrow);
