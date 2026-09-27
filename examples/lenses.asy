// The same colonnade photographed through three lenses with the camera
// module. Each camera stands at the distance that makes the sphere in the
// foreground fill the same part of its frame, so the sphere looks the same in
// every shot while the columns behind it do not: the wide-angle lens,
// close to the sphere, stretches the colonnade out into the distance, and the
// telephoto lens, far away, compresses it. The cameras are level, so the
// columns stay vertical; a lens shift raises the view to take in more of the
// colonnade.

import camera;
import graph3;

real width=6cm, height=4cm; // Size of each shot; 3:2, like 35mm film.
real[] focallengths={24,50,135};

// The default headlamp, under a pale blue sky.
currentlight=light(white,specular=gray(0.7),background=rgb(0.7,0.82,0.95),
                   specularfactor=3,dir(42,48));

// --- The scene ---------------------------------------------------------------

picture scene;

// A checkered floor, colored with a param pen, which is given the indices of
// each patch.
pen checker(pair uv, int U, int V) {
  return (U+V) % 2 == 0 ? gray(0.85) : rgb(0.35,0.4,0.5);
}
surface floor=surface(new triple(pair z) {return (z.x,z.y,0);},
                      (-6,-15),(6,45),12,60);
draw(scene,floor,parampen=checker);

// Two rows of columns receding from the sphere, each with a capital.
for(int i=1; i <= 15; ++i)
  for(int side : new int[] {-1,1}) {
    triple base=(2side,3i,0);
    draw(scene,shift(base)*scale(0.25,0.25,3)*unitcylinder,
         rgb(0.9,0.85,0.7));
    draw(scene,shift(base+(-0.4,-0.4,3))*scale(0.8,0.8,0.25)*unitcube,
         rgb(0.9,0.85,0.7));
  }

// The subject: a sphere on a pedestal.
draw(scene,shift(-0.3,-0.3,0)*scale(0.6,0.6,0.5)*unitcube,gray(0.5));
draw(scene,shift(0,0,1)*scale3(0.5)*unitsphere,red);

// --- The shots ---------------------------------------------------------------

triple target=(0,0,1);   // The center of the sphere, at eye level.
real halfwidth=1.6;      // Half the width of the view at the sphere.

// The scene through a lens of the given focal length (in mm, on 35mm film),
// from the distance at which the view at the sphere is 2*halfwidth wide.
frame shot(real focallength) {
  real fov=lens(focallength);
  real distance=halfwidth/Tan(fov/2);
  picture pic;
  size(pic,width,height);
  add(pic,scene);
  return pic.fit(P=camera(target-(0,distance,0),target,fov=fov,
                          aspect=width/height,shift=(0,0.2)));
}

for(int i=0; i < focallengths.length; ++i) {
  frame f=shot(focallengths[i]);
  pair z=(i*(width+0.3cm),0);
  // Place the bottom-left corner of the shot at z.
  add(f,z-min(f));
  draw(box(z,z+max(f)-min(f)),gray);
  label(format("%g mm",focallengths[i]),z+(width/2,0),S);
}
