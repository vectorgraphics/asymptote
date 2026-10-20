import TestLib;
import geometry;

// Regression test for issues #333 and #334:
// arc(ellipse, point, point) and arc(ellipse, real, real) must be interpreted
// with respect to the current coordinate system, not the default one.
// A circle has no distinguished major axis, so the spurious value that
// degrees(F2 - F1) (with F1 == F2) produced in a rotated coordinate system
// used to shift angle-based drawing (arc, angabscissa, ...) by the angle of
// the coordinate system's i-axis.
StartTest("coordsysarc");

pair[] endpoints(path p)
{
  return new pair[] {point(p, 0), point(p, length(p))};
}

void checkarc(path p, pair P, pair Q)
{
  // the drawn arc must connect P and Q (in either order)
  pair[] e = endpoints(p);
  assert((close(e[0], P) && close(e[1], Q))
      || (close(e[0], Q) && close(e[1], P)));
}

void run(string tag)
{
  coordsys R = currentcoordsys;

  // #334: arc between two explicit points on a circle
  point P = (0, 0), Q = (-5, 1);
  real v = abs(P - Q);
  point Rr = P + (0, v); // on the circle with center P and radius PQ
  checkarc((path)arc(circle(P, v), Q, Rr), locate(Q), locate(Rr));

  // #334 (second example): points taken with angpoint
  point P2 = (0, 0);
  point Q2 = angpoint(circle(P2, 5), 90);
  point R2 = angpoint(circle(P2, 5), 135);
  checkarc((path)arc(circle(P2, 5), Q2, R2), locate(Q2), locate(R2));

  // #333: arc angles measured from the i-axis of the current coordinate system
  point C = (1, 1);
  circle c = circle(C, 4);
  path ac = (path)arc(c, 45, 135);
  assert(close(point(ac, 0),
               locate(C) + (R * (4 * cos(radians(45)), 4 * sin(radians(45))))));
  assert(close(point(ac, length(ac)),
               locate(C) + (R * (4 * cos(radians(135)), 4 * sin(radians(135))))));

  // round-trip consistency: point(el, relabscissa(el, M)) == M
  ellipse el = circle(P, v);
  assert(close(locate(point(el, relabscissa(el, Q))), locate(Q)));
  assert(close(locate(point(el, relabscissa(el, Rr))), locate(Rr)));
}

run("default");

currentcoordsys = cartesiansystem(i = N, j = W);
run("rotated i=N j=W");

currentcoordsys = cartesiansystem(i = dir(30), j = dir(120));
run("rotated i=30 j=120");

EndTest();

