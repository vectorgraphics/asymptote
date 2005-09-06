import three;
import graph3;
size(200,0);
currentprojection=perspective(5,4,2);

real f(pair z) {return 0.5+exp(-abs(z)^2);}

draw((-1,-1,0)--(1,-1,0)--(1,1,0)--(-1,1,0)--cycle3);

draw(arc(0.12Z,0.2,90,60,90,15),ArcArrow);

real r=1.75;
draw(Label("$x$",1),O--r*X,red,Arrow);
draw(Label("$y$",1),O--r*Y,red,Arrow);
draw(Label("$z$",1),O--r*Z,red,Arrow);
label("$O$",(0,0,0),S,red);
  
draw((1,0,0)..(0,1,0)..(-1,0,0)..(0,-1,0)..cycle3,green);
  
add(surface(f,(-1,-1),(1,1),10,4));

