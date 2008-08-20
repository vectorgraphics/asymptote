import three;

currentprojection=orthographic(5,4,2);

size(5cm);
size3(3cm,5cm,8cm);

draw(unitcube);

dot(unitcube,red);

label("$O$",(0,0,0),NW);
label("(1,0,0)",(1,0,0),S);
label("(0,1,0)",(0,1,0),E);
label("(0,0,1)",(0,0,1),Z);
