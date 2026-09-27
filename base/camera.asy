// Place a viewframe() camera using the terms of photography: where the
// camera is, what it looks at, which way is up, and its field of view.
//
//   import camera;
//   size(400,300);
//   currentprojection=camera((6,-8,4),target=O,fov=50);
//
// The result is a viewframe() projection, so it supports only rendered
// (bitmap) output, and nothing nearer to the eye than the near clipping plane
// is drawn.

import three;

// The angle of view, in degrees, across a sensor (or film) of the given
// width behind a lens of the given focal length, both in millimeters. The
// default sensor width is that of 35mm film, so that, for instance,
// camera(eye,fov=lens(50)) behaves like a 50mm lens on a 35mm camera
// (measuring the angle across the longer side of the image).
real lens(real focallength, real sensor=36)
{
  if(focallength <= 0) abort("lens: focal length must be positive");
  if(sensor <= 0) abort("lens: sensor width must be positive");
  return 2aTan(sensor/(2focallength));
}

// The width/height ratio of pic, if its size has been set, and 1 otherwise.
private real aspect(picture pic)
{
  return pic.xsize > 0 && pic.ysize > 0 ? pic.xsize/pic.ysize : 1;
}

// A perspective camera at eye, looking toward target, turned about its line
// of sight so that up points as nearly upward in the image as possible.
// Additional arguments:
//
//   fov     The field of view, in degrees, strictly between 0 and 180.
//   fovaxis Which extent of the image fov measures: "horizontal",
//           "vertical", "diagonal", or "auto" (the longer side).
//   aspect  The width/height ratio of the image. Defaults to that of the
//           size of currentpicture, so call size(width,height) first (or
//           pass aspect); if the picture has no size, the default is 1.
//   roll    Starting from the orientation set by up, rolls the camera
//           about its line of sight so that the image turns roll degrees
//           counterclockwise, as in absperspective(). This is only a
//           convenience: roll=a is the same as
//           up=rotate(a,O,target-eye)*up.
//   shift   A lens shift: moves the image off the line of sight by
//           shift.x image widths rightward and shift.y image heights
//           upward, without turning the camera. For instance, a camera
//           aimed horizontally with shift=(0,0.3) photographs a tall
//           building without making its verticals converge.
//   focus   The distance from the eye, along the line of sight, at which
//           pen widths and labels have their nominal size in the image.
//           Defaults to the distance to target. (Without a call to size(),
//           the image is sized in bp as the view at this distance is in
//           user coordinates.)
//   near    The distance from the eye, along the line of sight, of the near
//           clipping plane; nothing nearer is drawn. Defaults to focus/100.
projection camera(triple eye, triple target=O, triple up=Z, real fov=50,
                  string fovaxis="auto", real aspect=aspect(currentpicture),
                  real roll=0, pair shift=(0,0), real focus=0, real near=0)
{
  triple forward=target-eye;
  if(forward == O) abort("camera: eye cannot be at target");
  real distance=abs(forward);
  forward /= distance;
  if(!(fov > 0 && fov < 180))
    abort("camera: fov must be strictly between 0 and 180 degrees");
  if(!(aspect > 0)) abort("camera: aspect must be positive");
  if(focus == 0) focus=distance;
  if(!(focus > 0)) abort("camera: focus must be positive");
  if(near == 0) near=focus/100;
  if(!(near > 0)) abort("camera: near must be positive");

  triple Up=up-dot(up,forward)*forward;
  if(abs(Up) <= sqrtEpsilon*abs(up))
    abort("camera: up must not be parallel to the line of sight");
  Up=rotate(roll,O,forward)*unit(Up);
  triple right=cross(forward,Up);

  if(fovaxis == "auto") fovaxis=aspect >= 1 ? "horizontal" : "vertical";
  real extent=2focus*Tan(fov/2);
  real width, height;
  if(fovaxis == "horizontal") {
    width=extent;
    height=extent/aspect;
  } else if(fovaxis == "vertical") {
    width=extent*aspect;
    height=extent;
  } else if(fovaxis == "diagonal") {
    real d=sqrt(1+aspect^2);
    width=extent*aspect/d;
    height=extent/d;
  } else
    abort("camera: fovaxis must be \"horizontal\", \"vertical\", "+
          "\"diagonal\", or \"auto\"");

  triple u=width*right, v=height*Up;
  triple center=eye+focus*forward+shift.x*u+shift.y*v;
  return viewframe(eye,center-u/2-v/2,u,v,near);
}
