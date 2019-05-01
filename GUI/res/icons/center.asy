// Empty bucket:  asy bucket -f svg
// Filled bucket: asy bucket -f svg -u fill=true -o filledbucket

real pixel=1inch/96;
size(25*pixel);
defaultpen(1.5bp);

draw(scale(2)*shift(-0.5,-0.5)*unitsquare);
fill(scale(0.5)*unitcircle);
