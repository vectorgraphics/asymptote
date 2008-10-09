size(100); 
import solids; 

currentprojection=orthographic(5,4,2);

revolution sphere=sphere(1); 
draw(surface(sphere),green+opacity(0.2));
draw(sphere,m=7,blue);

