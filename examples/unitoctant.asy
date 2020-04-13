import graph3;

currentprojection=orthographic(5,5,8);

size(0,150);
patch s0=octant1.s[0];
patch s1=octant1.s[1];
draw(surface(s0),green+opacity(0.5));
draw(surface(s1),green+opacity(0.5));
draw(s0.external(),blue);
draw(s1.external(),blue);

triple[][] P0=s0.P;
triple[][] P1=s1.P;

for(int i=0; i < 4; ++i)
  dot(P0[i],red+0.75mm);

for(int i=0; i < 4; ++i)
  dot(P1[i],red+0.65mm);

axes3("$x$","$y$",Label("$z$",align=Z));


