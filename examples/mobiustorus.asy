// A Mobius torus: a hexagonal tube bent into a ring whose cross-section turns
// by one sixth of a revolution on each trip around. The twist carries each face
// of the tube onto the next, so the six faces are really a single strip that
// winds around the ring six times before closing up. A param pen colors that
// strip continuously through the color wheel, returning to its starting color.

import graph3;
import palette;

size(12cm);
currentprojection=perspective(5,4,3);

real R=3;     // major radius
real a=3/4;   // circumradius of the hexagonal cross-section
int sides=6;  // sides of the cross-section; the twist turns it by one side

// Parametrize the whole strip at once. The parameter u (0..sides) counts laps
// around the ring, and v (0..1) runs across one face of the tube. Over one lap
// the cross-section turns by 2pi/sides, so the edge of the face at v=1 lies on
// top of the edge at v=0 one lap later, and after sides laps the strip returns
// to where it began. The surface is thus periodic in u but not in v.
triple mobius(pair uv) {
  real u=uv.x, v=uv.y;
  real theta=2pi*u;              // angle around the ring
  real phi=2pi*u/sides;          // twist of the cross-section
  // Point v of the way along the straight face between two adjacent vertices
  // of the hexagon, in the (radial, vertical) plane of the cross-section.
  pair p=a*interp(expi(phi),expi(phi+2pi/sides),v);
  triple radial=(cos(theta),sin(theta),0);
  return (R+p.x)*radial+p.y*Z;
}

// Spline detects that the data is periodic in u, making the strip cyclic in u
// so that it closes up seamlessly. Across each flat face a single patch
// suffices, since the face is straight in the v direction.
int nlap=32;                       // patches per lap around the ring
surface s=surface(mobius,(0,0),(sides,1),sides*nlap,1,Spline);

pen[] wheel=Wheel();               // rainbow palette running from red toward red
wheel.cyclic=true;                 // wrap indices around the end of the wheel

// Map a real, periodic with period 1, to the rainbow wheel.
pen spectrum(real t) {
  t %= 1.0;                        // wrap t into [0,1)
  t *= wheel.length;               // scale to the palette size
  int i=floor(t);
  return interp(wheel[i],wheel[i+1],t-i);
}

// The color depends only on how far along the strip we are: once around the
// color wheel over all sides laps, so the strip ends on the color it started.
pen strippen(pair uv, int, int) {
  return spectrum(uv.x/sides);
}

draw(s,parampen=strippen,nolight);
