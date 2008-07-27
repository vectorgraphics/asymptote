import three;

size(0,100);

currentprojection=obliqueX;

draw(unitcube);
dot(unitcube,red);

label("$O$",YZ,(0,0,0),NW);
label("(1,0,0)",YZ,(1,0,0),S);
label("(0,1,0)",YZ,(0,1,0),E);
label("(0,0,1)",YZ,(0,0,1),N);

