size(0,100);
import math;
import three;

currentprojection=oblique;

draw(unitcube);
dot(unitcube,red);

label("$O$",(0,0,0),NW);
label("(1,0,0)",(1,0,0),E);
label("(0,1,0)",(0,1,0),N);
label("(0,0,1)",(0,0,1),S);
