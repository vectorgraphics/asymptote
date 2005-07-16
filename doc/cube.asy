size(0,100);
import math;
import three;

currentprojection=oblique;

path3[] p=box3d((0,0,0),(1,1,1));
draw(p);
dot(p,red);

label("O",(0,0,0),NW);
label("(1,0,0)",(1,0,0),E);
label("(0,1,0)",(0,1,0),N);
label("(0,0,1)",(0,0,1),S);
