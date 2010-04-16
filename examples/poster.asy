orientation=Landscape;
import slide;
import graph;

defaultpen(deepblue);
pagenumberpen=invisible;

real f(real x) {return (x != 0) ? x*sin(1/x) : 0;}
pair F(real x) {return (x,f(x));}

xaxis(background,grey);
yaxis(background,-0.25,0.25,grey);
real a=1.2/pi;
draw(background,graph(background,f,-a,a,10000),grey);
label(background,"$x\sin\frac{1}{x}$",F(0.92/pi),3SE,grey+fontsize(14pt));
frame f=background.fit();
box(f,RadialShade(yellow,0.6*yellow+red),above=false);
background.erase();
add(background,f);  

title("Young Researchers' Conference",align=3S,fontsize(48pt));
center("University of Alberta, Edmonton, April 1--2, 2006");

skip(4);

center("A general conference for\\
the mathematical and statistical sciences\\
for graduate students, by graduate students.",fontsize(32pt));

label("Registration and abstract submission online.",(0,-0.5));

label("\tt http://www.pims.math.ca/science/2006/06yrc/",point(SW),2NE,
      black+fontsize(18pt));
