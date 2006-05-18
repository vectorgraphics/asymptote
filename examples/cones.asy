size(200);
import solids;

currentprojection=orthographic(5,4,2);

revolution cone=revolution(-X-Z--O--X+Z,Z);
cone.filldraw(green,5,blue);

revolution cone2=shift(2Y-2X)*revolution(O--X+Z,Z);
cone2.filldraw(green,blue);

