import graph;

real r=4;
real h=10;
real s=8;

pair z0=(0,0);
pair z1=(r,h);
pair z2=(-r,h);

draw(z0--z1--z2--cycle);
real x=r*s/h;
guide g=(0,0)--(-x,s)--(x,s)--cycle;
fill(g,lightblue+white);
yaxis(0,dotted);
xaxis();
draw(g);

real s1=5.0;
real s2=5.1;
real x1=r*s1/h;
real x2=r*s2/h;
guide g2=(-x2,s2)--(-x1,s1)--(x1,s1)--(x2,s2)--cycle;
fill(g2,blue);

draw((string) s,(-x,0)--(-x,s),W,red,Arrow,Bar,PenMargin);
draw("$x$",(x2,0)--(x2,0.5*(s1+s2)),red,Arrow,Bar,PenMargin);
draw((string) h,(r,0)--(r,h),red,Arrow,Bar,PenMargin);
label("$r$",(0.5*x2,s2),0.5*N,red);

label((string) r,(0.5*r,h+0.3),0.5*N,red);

shipout(0,150);
