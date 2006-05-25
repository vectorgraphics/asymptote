size(200);
import solids;

currentprojection=perspective(5,4,4);

revolution r=revolution(shift(3X)*Circle(O,1,Y,32),Z,90,345);
r.fill(green);
