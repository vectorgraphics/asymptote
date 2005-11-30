size(400);

pen convex=makepen(scale(10)*polygon(8));
draw((0.2,0),convex);
draw((0,0)---(1,1)..(2,0),convex);

pen nonconvex=scale(10)*
  makepen((0,0)--(0.25,-1)--(0.5,0.25)--(1,0)--(0.5,1.25)--cycle)+red;

draw((0.2,-1),nonconvex);
draw((0,-1)..(1,0)..(2,-1),nonconvex);
