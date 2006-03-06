import graph;
import fontsize;
size(11inches,8inches,IgnoreAspect);

defaultpen(blue+0.5Black+fontsize(20));

real f(real x) {return (x != 0) ? x*sin(1/x) : 0;}
pair F(real x) {return (x,f(x));}

xaxis(grey);
yaxis(grey);
real a=1.2/pi;
draw(graph(f,-a,a,10000),grey);
label("$x\sin\frac{1}{x}$",F(0.92/pi),3SE,grey+fontsize(14));

label("Young Researchers' Conference",point(N),3S,fontsize(48));
label("University of Alberta, Edmonton, April 1--2, 2006",(0,0.1));

label(minipage("\center{A general conference for\\
the mathematical and statistical sciences\\
for graduate students, by graduate students.}",25cm),(0,0.02),fontsize(40));
label("Registration and abstract submission online.",(0,-0.15));

label("\tt http://www.pims.math.ca/science/2006/06yrc/",1.1*point(S),W,
      black+fontsize(18));

shipout(Landscape(bbox(RadialShade(yellow,0.6*yellow+red))));
