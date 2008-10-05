import solids;

size(200);
currentprojection=orthographic(5,4,2);

revolution upcone=cone(-Z,1,1);
revolution downcone=cone(Z,1,-1);
draw(surface(upcone),green);
draw(surface(downcone),green);
draw(upcone,5,blue);
draw(downcone,5,blue);

revolution cone=shift(2Y-2X)*cone(1,1);

draw(surface(cone),green);
draw(cone,5,blue);
