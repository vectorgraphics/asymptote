size(6cm,0);
import math;
import three;

real u=2.5;
real v=1;

currentprojection=oblique;

path3 y=plane((2u,0,0),(0,2v,0),(-u,-v,0));
path3 l=rotate(90,Z)*rotate(90,Y)*y;
path3 g=rotate(90,X)*rotate(90,Y)*y;

face[] faces;
filldraw(faces.push(y),y,yellow);
filldraw(faces.push(l),l,lightgrey);
filldraw(faces.push(g),g,green);

add(faces);

