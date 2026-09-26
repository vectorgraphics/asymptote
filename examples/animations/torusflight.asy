// A flight around and through a torus, filmed with viewframe() projections.
// The camera approaches from far away, skims once around the outside of the
// torus along a (1,1) curve, which winds once around the central hole and
// once around the tube, with "up" pointing away from the surface, like a
// trench run. It then dives through the surface and flies the same loop
// inside the tube. Nothing between the eye and the picture plane is drawn,
// so the dive cuts a window through the hull.
//
// Rendering every frame takes a while. To render just the frame at time t
// (in seconds), run
//   asy -f png -u "still=t" torusflight
// and to make the animation (a GIF, unless another format is given with -f),
//   asy torusflight

import graph3;
import palette;
import animation;

real still=-1;    // If nonnegative, render only the frame at this time.
usersetting();

int width=480, height=270; // Image size in bp.
real fps=24;               // Frames per second.

// --- The torus ---------------------------------------------------------------

real R=3; // major radius
real a=1; // tube radius

// The point at distance rho from the core circle of the tube, in the
// direction normal(u,v); for rho=a it is the point of the torus at parameters
// (u,v), where u runs around the central hole and v around the tube.
triple normal(real u, real v) {return (cos(v)*cos(u),cos(v)*sin(u),sin(v));}
triple torus(real u, real v, real rho=a) {
  return R*(cos(u),sin(u),0)+rho*normal(u,v);
}

int nloop=160, ntube=64; // Patches around the hole and around the tube.
surface s=surface(new triple(pair uv) {return torus(uv.x,uv.y);},
                  (0,0),(2pi,2pi),nloop,ntube,Spline);

// --- The hull pattern --------------------------------------------------------

pen[] wheel=Wheel(); // Cyclic rainbow palette.
wheel.cyclic=true;

// Map a real, periodic with period 1, to the rainbow wheel.
pen spectrum(real t) {
  t=t-floor(t);
  t *= wheel.length;
  int i=floor(t);
  return interp(wheel[i],wheel[i+1],t-i);
}

// A pseudorandom number in [0,1) determined by the indices of a patch.
real hash(int U, int V) {
  real x=sin(12.9898*U+78.233*V)*43758.5453;
  return x-floor(x);
}

// The flight path runs along the (1,1) curve v=u+phase, starting on top of
// the torus.
real phase=pi/2;

// Steel plating whose panels (patches) vary in shade, with a few lit windows,
// and a neon trench along the flight path whose hue cycles once around the
// loop. The plating depends only on the patch indices, so each panel is flat;
// the trench depends on the parameters at each corner, so it is smooth.
pen hull(pair uv, int U, int V) {
  real h=hash(U,V);
  pen plate=h < 0.04 ? rgb(1,0.8,0.35) :
    gray(0.3+0.08*((U+V)%2)+0.12*h);
  // Signed distance around the tube from the center line of the trench.
  real d=uv.y-uv.x-phase;
  d -= 2pi*round(d/(2pi));
  real w=0.1;
  real glow=max(1-abs(d)/w,0);
  return interp(plate,spectrum(uv.x/(2pi)),glow);
}

// A field of stars: small balls scattered over a distant sphere. (Dots would
// not do, since their size is given in bp at the picture plane, which lies
// very close to the eye.)
srand(1);
surface stars;
for(int i=0; i < 400; ++i) {
  real z=2unitrand()-1;
  real phi=2pi*unitrand();
  real r=sqrt(1-z^2);
  stars.append(shift(80*(r*cos(phi),r*sin(phi),z))*scale3(0.3)*unitsphere);
}

// --- The flight path ---------------------------------------------------------

real hOut=0.12;          // Height of the eye above the hull, outside.
real rhoIn=0.45;         // Distance of the eye from the core, inside.
real dive=pi;            // Loop parameter spent diving through the hull.
real speed=3;            // Speed along the loops, in units per second.
real approachTime=4;     // Seconds spent approaching.
real pitchOut=12;        // Degrees by which the camera looks down, outside;
real pitchIn=-6;         // and inside, where it looks up at the trench.

// The loop parameter theta runs from 0 to 2pi around the outside, then over
// the dive, and then through 2pi more around the inside. Return how far the
// dive has progressed at theta, from 0 (outside) to 1 (inside).
real inside(real theta) {
  real x=min(max((theta-2pi)/dive,0),1);
  return x*x*(3-2x); // smoothstep
}
real thetaEnd=4pi+dive;

// Position, up direction, and pitch at loop parameter theta. The eye is at
// distance rho from the core circle of the tube.
triple eye(real theta) {
  real rho=interp(a+hOut,rhoIn,inside(theta));
  return torus(theta,phase+theta,rho);
}
triple up(real theta) {return normal(theta,phase+theta);}
real pitch(real theta) {return interp(pitchOut,pitchIn,inside(theta));}

// The approach: a Bezier curve from far away, initially heading for the
// center of the torus, that joins the loop tangentially, with up=Z, which is
// also the up direction where it joins.
triple E0=eye(0);
triple f0=unit(eye(1e-4)-eye(-1e-4));
triple far=E0-30f0+(0,0,12);
path3 approach=far..controls (far+15unit(O-far)) and (E0-12f0)..E0;

// Sample the whole flight densely, recording position, up direction, and
// pitch, then compute cumulative arc length so the camera can move at a
// controlled speed.
triple[] position, upward;
real[] tilt;
int nApproach=600, nLoop=6000;
for(int i=0; i < nApproach; ++i) {
  position.push(point(approach,i/nApproach));
  upward.push(Z);
  tilt.push(pitchOut);
}
for(int i=0; i <= nLoop; ++i) {
  real theta=thetaEnd*i/nLoop;
  position.push(eye(theta));
  upward.push(up(theta));
  tilt.push(pitch(theta));
}
real[] length={0};
for(int i=1; i < position.length; ++i)
  length.push(length[i-1]+abs(position[i]-position[i-1]));
real approachLength=length[nApproach];
real loopLength=length[length.length-1]-approachLength;
real duration=approachTime+loopLength/speed;

// Distance flown by time t: the approach starts from rest and arrives at the
// loop speed; thereafter the speed is constant.
real distance(real t) {
  if(t >= approachTime) return approachLength+speed*(t-approachTime);
  real x=t/approachTime;
  real m=speed*approachTime/approachLength; // Final slope, relative.
  return approachLength*((3-2x)*x^2+m*(x-1)*x^2);
}

real planeDistance=0.05;     // Distance from the eye to the picture plane.
real fieldOfView=80;         // Horizontal field of view, in degrees.
triple sun=unit((-1,-2,3));  // Direction to the sun, fixed in the world.

// Place the camera for time t, setting currentprojection and currentlight.
void camera(real t) {
  // Interpolate the samples at the distance flown.
  real l=min(max(distance(t),0),length[length.length-1]);
  int i=min(search(length,l),length.length-2);
  real x=(l-length[i])/(length[i+1]-length[i]);
  triple E=interp(position[i],position[i+1],x);
  // Look along the flight path, pitched down toward the hull outside and up
  // toward the trench inside.
  triple forward=unit(position[i+1]-position[i]);
  triple Up=interp(upward[i],upward[i+1],x);
  Up=unit(Up-dot(Up,forward)*forward);
  triple right=cross(forward,Up);
  forward=rotate(-interp(tilt[i],tilt[i+1],x),right)*forward;
  Up=cross(right,forward);

  real w=2*planeDistance*Tan(fieldOfView/2);
  real h=w*height/width;
  triple u=w*right, v=h*Up;
  triple corner=E+planeDistance*forward-u/2-v/2;
  currentprojection=viewframe(E,corner,u,v);

  // The renderer takes light directions in eye coordinates. Besides the
  // sun, fill lights that move with the camera, from ahead, above, and below,
  // keep the hull visible where the sun cannot reach it.
  transform3 T=shiftless(currentprojection.T.modelview);
  currentlight=light(new pen[] {gray(0.6),gray(0.3),gray(0.35),gray(0.2)},
                     new pen[] {gray(0.6),black,black,black},background=black,
                     new triple[] {T*sun,(0,0,1),(0,1,0.3),(0,-1,0.3)});
}

// The picture seen at time t.
picture shot(real t) {
  picture pic;
  size(pic,width,height);
  draw(pic,s,parampen=hull);
  draw(pic,stars,white,nolight);
  camera(t);
  return pic;
}

if(still >= 0)
  currentpicture=shot(still);
else {
  animation A=animation(global=false);
  int n=ceil(duration*fps);
  // Fill the margin that the frames acquire when they are embedded with the
  // background color.
  for(int i=0; i < n; ++i)
    A.add(shot(i/fps),BBox(1,black,Fill(black)));
  A.movie(delay=1000/fps);
}
