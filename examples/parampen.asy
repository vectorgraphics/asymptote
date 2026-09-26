// Two interlocked tori colored at drawing time with a param pen, which is
// evaluated at the parametric coordinates of each patch corner. One torus
// carries a smooth spiral of rainbow color, the other a flat checkerboard.

import graph3;
import palette;

size(12cm);
currentprojection=perspective(5,4,3);
currentlight=Viewport;

real R=3;   // major radius
real a=1;   // minor (tube) radius

// A torus as a parametric surface. The point at parameter (u,v) is displaced
// from the major circle (radius R, swept by u) by a*cos(v) radially outward
// from the central axis and a*sin(v) vertically along that axis: u (radians,
// 0..2pi) runs around the major loop and v (radians, 0..2pi) around the tube.
triple torus(pair uv) {
  real u=uv.x, v=uv.y;
  return R*(cos(u), sin(u), 0)        // point on the major circle
       + a*cos(v)*(cos(u), sin(u), 0)   // radial, outward from the central axis
       + a*sin(v)*(0, 0, 1);            // vertical, along the symmetry axis
}

// Build the torus at the given patch resolution. Spline auto-detects that the
// data is periodic in both u and v, so the surface is cyclic in both directions
// and closes up seamlessly. Because the surface is parametric, the (u,v) it
// records are exactly the radians fed to the formula above, and those same
// (u,v) are what the param pen receives -- the pen and the surface share one
// coordinate system.
surface torus(int nloop, int ntube) {
  return surface(torus, (0,0), (2pi,2pi), nloop, ntube, Spline);
}

// --- Spiral torus -----------------------------------------------------------

int nloop=40, ntube=20;             // patch resolution of the spiral torus
pen[] wheel=Wheel();                // cyclic rainbow palette (red back to red)
wheel.cyclic=true;                   // wrap indices around the end of the wheel

// Map a real, periodic with period 1, to the rainbow wheel.
pen spectrum(real t) {
  t %= 1.0;  // wrap t into [0,1)
  t *= wheel.length;  // scale to the palette size
  return interp(wheel[floor(t)], wheel[ceil(t)], t % 1.0);
}

// Winding numbers: how many times the hue cycles around the major loop and
// around the tube. They MUST be integers so the spiral closes up seamlessly:
// advancing u by 2pi or v by 2pi then shifts the argument by a whole number,
// leaving the color unchanged (spectrum has period 1). The pen reads the
// surface's own (u,v) directly -- no remapping from indices needed.
int windLoop=3, windTube=1;
pen spiralpen(pair uv, int, int) {
  real loop=uv.x/(2pi);             // 0..1 around the major loop
  real tube=uv.y/(2pi);             // 0..1 around the tube
  return spectrum(windLoop*loop+windTube*tube);
}

draw(torus(nloop,ntube),parampen=spiralpen);

// --- Checkerboard torus -----------------------------------------------------

// A coarser, even resolution so the checkerboard tiles the surface evenly and
// its parity also matches across the two cyclic seams.
int mloop=24, mtube=12;
pen cream=rgb(0.95,0.93,0.86);
pen navy=rgb(0.12,0.18,0.45);

// The checkerboard depends only on the patch's u,v indices in the index grid,
// so all four corners of a patch share one color: crisp, flat squares.
pen checkerpen(pair, int U, int V) {
  return (U+V)%2 == 0 ? cream : navy;
}

// Link the second torus through the first: rotate it into the xz plane and
// shift its center onto the first torus's major circle, so it threads the hole.
draw(shift((R,0,0))*rotate(90,X)*torus(mloop,mtube), parampen=checkerpen);
