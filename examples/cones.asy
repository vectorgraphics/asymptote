size(200);
import solids;

currentprojection=orthographic(5,4,2);

revolution cones=revolution(-X-Z--O--X+Z,Z);
draw(surface(cones),green);
draw(cones,5,blue);

revolution cone=shift(2Y-2X)*cone(1,1);
draw(surface(cone),green);
draw(cone,5,blue);
