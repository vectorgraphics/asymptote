import graph;
size(0,100);

pair z0=(0,0);
real r=1;
real a1=45;
real a2=300;
guide g=arc(z0,r,a1,a2);
draw(g);
pair z1=point(g,0);
pair z2=point(g,length(g));

real r2=1.1*r;
guide c=arc(0,r2,a1,a2);
draw("$2\pi r$",c,red,Arrows,Bars);
draw("$\ell$",z0--z1,0.5*SE,blue);
draw(z0--z2);
draw(arc(z0,r,a2-360,a1),dashed+green);

shipout();
