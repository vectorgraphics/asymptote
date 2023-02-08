import graph3;
size(300);

pen yellow=rgb("F1FA8C");
pen purple=rgb("BB95FF");
pen blue=rgb("7A8FFE");
pen darkblack=rgb("101010");

currentprojection=orthographic(3,3,1,up=Z);
currentlight=light((1,0,1),(-1.5,0,-1));
currentlight.background=darkblack;

real aS=2.5;
draw(Label("$x$",EndPoint),-aS*X--aS*X,white,Arrow3);
draw(Label("$y$",EndPoint),-aS*Y--aS*Y,white,Arrow3);
draw(Label("$z$",EndPoint),-aS*Z--aS*Z,white,Arrow3);

draw(shift(0.5,0,-2)*scale(0.5,0.5,4)*unitcylinder,
     material(blue+opacity(0.8),shininess=0.3));

draw(unitsphere,material(purple,shininess=0.3));

triple f(real t) {return(cos(t)^2,cos(t)*sin(t),sin(t));}

path3 curve=graph(f,0,8pi,operator ..);
draw(curve,yellow+linewidth(1));
