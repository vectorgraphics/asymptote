import graph;
size(0,100);

real r=1;
real h=3;

yaxis(dashed);

real m=0.475*h;

draw((r,0)--(r,h));
label("$L$",(r,0.5*h),E);

real s=4;

pair z1=(s,0);
pair z2=z1+(2*pi*r,h);
filldraw(box(z1,z2),lightgreen);
pair zm=0.5*(z1+z2);
label("$L$",(z1.x,zm.y),W);
label("$2\pi r$",(zm.x,z2.y),N);
draw("$r$",(0,m)--(r,m),N,red,Arrows);

draw((0,1.015h),yscale(0.5)*arc(0,0.25cm,-250,70),red,ArcArrow);


