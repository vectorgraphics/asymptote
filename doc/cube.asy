import three;

size(100);

currentprojection=obliqueX;

draw(unitcube);
dot(unitcube,red);

label("$O$",(0,0,0),NW);
label("(1,0,0)",(1,0,0),S);
label("(0,1,0)",(0,1,0),E);
label("(0,0,1)",(0,0,1),N);
