import solids;

size(200);
currentprojection=orthographic(5,4,2);

render render=render(compression=Low,merge=true);

revolution upcone=cone(-Z,1,1);
revolution downcone=cone(Z,1,-1);
draw(surface(upcone),green,render);
draw(surface(downcone),green,render);
draw(upcone,5,blue,longitudinalpen=nullpen);
draw(downcone,5,blue,longitudinalpen=nullpen);

revolution cone=shift(2Y-2X)*cone(1,1);

draw(surface(cone),green,render);
draw(cone,5,blue);
