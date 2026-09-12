// Sweep a plane through a nearly transparent cube, cropping the cube's
// surface to the near side of it. The rim where the cropped surface stops is
// the cross section, and it changes shape as the plane crosses each corner:
// over the whole animation it is a triangle, a quadrilateral, a pentagon and
// a hexagon in turn.
import three;
import crop3;
import animation;

size(8cm);
currentprojection=orthographic((3.6,2.2,1.9));
currentlight=Headlamp;

surface cube=shift(-0.5*(X+Y+Z))*unitcube;

// The cross section: those sides of the cropped surface that lie in the
// cutting plane. Each face the plane meets contributes one.
path3[] section(surface s, triple normal, real offset)
{
  path3[] p;
  for(patch q : s.s) {
    path3 g=q.external();
    for(int i=0; i < length(g); ++i)
      if(abs(dot(point(g,i),normal)-offset) < 1e-6 &&
         abs(dot(point(g,i+1),normal)-offset) < 1e-6)
        p.push(subpath(g,i,i+1));
  }
  return p;
}

animation a;

// u=0 puts the plane just past the corner of the cube nearest -normal and
// u=1 just past the opposite one, so one sweep covers the cube whatever the
// normal is.
void snapshot(triple normal, real u)
{
  real halfwidth=0.5*(abs(normal.x)+abs(normal.y)+abs(normal.z));
  real offset=interp(-halfwidth,halfwidth,u);
  surface kept=crop(cube,normal,'<=',offset);
  path3[] rim=section(kept,normal,offset);

  picture pic;
  size(pic,8cm);
  draw(pic,cube,gray(0.5)+opacity(0.06),meshpen=gray(0.6)+0.2pt);
  draw(pic,kept,lightblue+opacity(0.9),meshpen=black+0.4pt);
  draw(pic,rim,red+1.4pt);
  label(pic,(string) rim.length+" edges",(0,0,1),N);
  a.add(pic);
}

int n=24;
triple upright=(1,1,1), tilted=(1,1,3);

// Corner to corner along a diagonal: triangle, hexagon, triangle.
for(int i=0; i <= n; ++i)
  snapshot(upright,interp(0.02,0.98,i/n));

// Tilt the plane while it is parked past the far corner, where the section
// stays a triangle, so that the tilt itself changes nothing.
for(int i=1; i <= quotient(n,4); ++i)
  snapshot(interp(upright,tilted,i/quotient(n,4)),0.98);

// Back again, now meeting the faces unevenly: triangle, pentagon,
// quadrilateral, pentagon, triangle.
for(int i=1; i <= n; ++i)
  snapshot(tilted,interp(0.98,0.02,i/n));

a.movie(loops=0,delay=100);
