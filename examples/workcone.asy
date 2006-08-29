import graph3;
import solids;
size(0,150);
currentprojection=orthographic(0,-30,5);

real r=4;
real h=10;
real s=8;
real x=r*s/h;

real sr=5;
real xr=r*sr/h;

real s1=sr-0.2;
real x1=r*s1/h;

real s2=sr+0.2;
real x2=r*s2/h;

guide3 p=(0,0,0)--(x,0,s);
revolution a=revolution(p,Z);
a.filldraw(lightblue,lightblue+white,false);

guide3 q=(x,0,s)--(r,0,h);
revolution b=revolution(q,Z);
b.filldraw(white,black,false);

bbox3 b=autolimits(O,h*(X+Z)+Y);

draw((-r-1,0,0)--(r+1,0,0));
draw((0,0,0)--(0,0,h+1),dashed);

guide3 w=(x1,0,s1)--(x2,0,s2)--(0,0,s2);
revolution b=revolution(w,Z);
b.filldraw(blue,black,false);
draw(circle((0,0,s2),x2));

draw("$x$",(xr,0,0)--(xr,0,sr-0.1),red,Arrow,Bar,PenMargin);
draw("$r$",(0,0,s2)--(-x2,0,s2),N,red);
draw((string) r,(0,0,h)--(r,0,h),N);
draw((string) h,(r,0,0)--(r,0,h),red,Arrow,Bar,PenMargin);
draw((string) s,(-x,0,0)--(-x,0,s),W,red,Arrow,Bar,PenMargin);
