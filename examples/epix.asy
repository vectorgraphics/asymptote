import graph3;

size(200,200,IgnoreAspect);

currentprojection=perspective((4,2,3));

real f(pair z) {return z.y^3/2-3z.x^2*z.y;}

add(surface(f,(-1,-1),(0,1),n=10));
draw(Label("$y$",1),(0,0,0)--(0,2,0),red,Arrow);
add(surface(f,(0,-1),(1,1),n=10));

draw(Label("$x$",1),(0,0,0)--(2,0,0),red,Arrow);
draw(Label("$z$",1),(0,0,0)--(0,0,2.5),red,Arrow);
