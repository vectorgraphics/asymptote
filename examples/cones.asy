size(200);
import solids;

currentprojection=orthographic(5,4,2);

revolution cones=revolution(-X-Z--O--X+Z,Z);
cones.filldraw(green,5,blue);

(shift(2Y-2X)*cone(1,1)).filldraw(green,5,blue);

