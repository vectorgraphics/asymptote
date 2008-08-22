size(200);
import solids;

currentprojection=perspective(5,4,4);

revolution torus=revolution(shift(3X)*Circle(O,1,Y,32),Z,90,345);
draw(surface(torus),green);
