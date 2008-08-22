size(200);
import solids;
currentprojection=orthographic(5,4,3);

revolution r=sphere(O,1);
draw(surface(r),green);
draw(r,3,blue);
