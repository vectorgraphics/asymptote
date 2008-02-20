size(0,150);

pair z0=(0,0);
real r=1;
real h=1;
real l=sqrt(r^2+h^2);
real a=(1-r/l)*360;
real a1=a/2;
real a2=360-a/2;
path g=arc(z0,r,a1,a2);
fill((0,0)--g--cycle,lightgreen);
draw(g);
pair z1=point(g,0);
pair z2=point(g,length(g));

real r2=1.1*r;
path c=arc(0,r2,a1,a2);
draw("$2\pi r$",c,red,Arrows,Bars,PenMargins);
pen edge=blue+0.5mm;
draw("$\ell$",z0--z1,0.5*SE,edge);
draw(z0--z2,edge);
draw(arc(z0,r,a2-360,a1),grey+dashed);
dot(0);
