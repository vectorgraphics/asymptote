import TestLib;
import graph3;

// Tests for the detection of cyclic parameters by the parametric surface
// constructor, surface(f, a, b, nu, nv, Spline).

triple tube(pair t) { return (cos(t.x), sin(t.x), t.y); }

StartTest("a tube two patches around is u-cyclic");
{
  // Sampled at u = 0, pi, 2pi, the y component sin(u) vanishes in exact
  // arithmetic, leaving only roundoff; periodicity must be judged against
  // the size of the whole sample, not that of the roundoff.
  surface s = surface(tube, (0,0), (2pi,1), 2, 2, Spline);
  assert(s.ucyclic(), "the tube closes up in u");
  assert(!s.vcyclic(), "the tube is open in v");
}
EndTest();

StartTest("a tube two patches around is v-cyclic");
{
  triple f(pair t) { return tube((t.y, t.x)); }
  surface s = surface(f, (0,0), (1,2pi), 2, 2, Spline);
  assert(s.vcyclic(), "the tube closes up in v");
  assert(!s.ucyclic(), "the tube is open in u");
}
EndTest();

StartTest("detection does not depend on the coordinate frame");
{
  for (real angle : new real[] {0, 30, 45, 90}) {
    triple f(pair t) { return rotate(angle, Z)*tube(t); }
    for (int nu : new int[] {2, 3, 8}) {
      surface s = surface(f, (0,0), (2pi,1), nu, 2, Spline);
      assert(s.ucyclic(), "the rotated tube closes up in u");
    }
  }
}
EndTest();

StartTest("surfaces that do not close up are not cyclic");
{
  // Half a tube returns to neither its starting point nor its slope.
  surface half = surface(tube, (0,0), (pi,1), 2, 2, Spline);
  assert(!half.ucyclic(), "half a tube is open");
  // One turn of a helicoid repeats after a translation, which suffices for
  // periodicOffset interpolation but not for u to be cyclic.
  triple helicoid(pair t) { return (t.y*cos(t.x), t.y*sin(t.x), t.x); }
  for (int nu : new int[] {2, 8}) {
    surface s = surface(helicoid, (0,0.5), (2pi,1), nu, 2, Spline);
    assert(!s.ucyclic(), "a helicoid does not close up");
  }
}
EndTest();
