import graph3;

size(200,200,IgnoreAspect);

currentprojection=perspective(4,2,3);

real f(pair z) {return z.y^3/2-3z.x^2*z.y;}

draw(surface(f,(-1,-1),(1,1),nx=10,Spline),green,render(merge=true));
draw(Label("$y$",1),(0,0,0)--(0,2,0),red,Arrow3);

draw(Label("$x$",1),(0,0,0)--(2,0,0),red,Arrow3);
draw(Label("$z$",1),(0,0,0)--(0,0,2.5),red,Arrow3);
label("$z=\frac{1}{2}y^3-3x^2y$",(1,1,1),NE);
