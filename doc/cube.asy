import three;

size(100);

currentprojection=obliqueX;

draw(unitcube);
dot(unitcube,red);

label(YZ*"$O$",(0,0,0),NW);
label(YZ*"(1,0,0)",(1,0,0),S);
label(YZ*"(0,1,0)",(0,1,0),E);
label(YZ*"(0,0,1)",(0,0,1),N);
