// Empty bucket:  asy bucket -f svg
// Filled bucket: asy bucket -f svg -u fill=true -o filledbucket

defaultpen(3.5);

real h=4;
real r=3;
path left=(-r,h)--(-r,0);
path right=(r,0)--(r,h);
path bottom=xscale(r)*arc(0,1,180,360);

real H=0.9h;
path Left=(-r,H/2)--(-r,0);
path Right=(r,0)--(r,H/2);

bool fill=false;    // Set to true for filled bucket.
usersetting();

if(fill)
  fill(Left--bottom--Right--shift(0,H)*xscale(r)*arc(0,1,0,180)--cycle,gray);

draw(shift(0,h)*xscale(r)*unitcircle);
draw(left--bottom--right);
draw(shift(0,h)*scale(r)*arc(0,1,0,180));

shipout(pad(64,64));
