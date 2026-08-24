import graph3;

currentprojection=orthographic(5,5,8);

size(0,150);
draw(unitoctant,green+opacity(0.5));
draw(octant1x.external(),blue);

axes3("$x$","$y$",Label("$z$",align=Z));
