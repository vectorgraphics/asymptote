import three;
size(200,0);

currentprojection=perspective((5,4,2));

real f(pair z) {return 0.5+exp(-abs(z)^2);}

draw((-1,-1,0)--(1,-1,0)--(1,1,0)--(-1,1,0)--cycle3);

real r=1.75;
draw(Label("$x$",1),(0,0,0)--(r,0,0),red,Arrow);
draw(Label("$y$",1),(0,0,0)--(0,r,0),red,Arrow);
draw(Label("$z$",1),(0,0,0)--(0,0,r),red,Arrow);
  
draw((1,0,0)..(0,1,0)..(-1,0,0)..(0,-1,0)..cycle3,green);

label("$O$",(0,0,0),S,red);
  
add(surface(f,(-1,-1),(1,1),n=10));

draw(arc(0.12Z,0.2,90,60,90,15),ArcArrow);
