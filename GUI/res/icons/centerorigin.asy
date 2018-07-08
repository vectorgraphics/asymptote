real pixel=1inch/96;
size(25*pixel);
defaultpen(1.5bp);

draw(scale(2)*shift(-0.5,-0.5)*unitsquare);
draw((-1,0)--(1,0));
draw((0,-1)--(0,1));
