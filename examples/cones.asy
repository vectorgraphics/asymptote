import solids;

size(200);
currentprojection=orthographic(5,4,2);

render render=render(compression=Low,merge=true);
pen skeletonpen=blue+0.15mm;

revolution upcone=cone(-Z,1,1);
revolution downcone=cone(Z,1,-1);
draw(surface(upcone),green,render);
draw(surface(downcone),green,render);
draw(upcone,5,skeletonpen,longitudinalpen=nullpen);
draw(downcone,5,skeletonpen,longitudinalpen=nullpen);

revolution cone=shift(2Y-2X)*cone(1,1);

draw(surface(cone),green,render);
draw(cone,5,skeletonpen);
