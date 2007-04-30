import graph3;

size(200,0);

currentprojection=perspective(5,4,2);

real f(pair z) {return 0.5+exp(-abs(z)^2);}

draw((-1,-1,0)--(1,-1,0)--(1,1,0)--(-1,1,0)--cycle);

draw(arc(0.12Z,0.2,90,60,90,15),ArcArrow);

picture surface=surface(f,nsub=4,(-1,-1),(1,1),nx=10,light=O);
  
bbox3 b=limits(O,1.75(1,1,1));

xaxis(Label("$x$",1),b,red,Arrow);
yaxis(Label("$y$",1),b,red,Arrow);
zaxis(Label("$z$",1),b,red,Arrow);

label("$O$",(0,0,0),S,red);
  
add(surface);

