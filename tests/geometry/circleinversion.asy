
import TestLib;
import geometry;

// Regression test for issue #143: inversion(circle1, circle2) must return
// an inversion which actually transforms the first circle into the second in
// every configuration (one containing the other, disjoint, intersecting,
// externally or internally tangent, concentric, equal radii), and the sgn
// argument must select between the two possible inversions by the location
// of the center (between the centers, or outside the segment joining them).
StartTest("circleinversion");

bool nearly(real a, real b)
{
  return abs(a - b) <= 100 * realEpsilon * max(abs(a), abs(b));
}

void checkinversion(circle c1, circle c2, real s, bool fallback = false)
{
  inversion iota = inversion(c1, c2, s);
  circle img = iota * c1;
  assert(close(locate(img.C), locate(c2.C)));
  assert(nearly(img.r, c2.r));
  // the two inversions are distinguished by their center: sgn > 0 gives the
  // center located between the centers, sgn < 0 the center located outside
  // the segment joining the centers (except when the centers coincide).
  // When fallback is true the preferred inversion does not exist and the
  // other one is returned, so the center is on the opposite side.
  pair A = locate(c1.C), B = locate(c2.C), O = locate(iota.C);
  if (A != B)
  {
    bool between = nearly(abs(A - O) + abs(O - B), abs(A - B));
    bool expectBetween = (s > 0) != fallback;
    assert(between == expectBetween);
  }
}

void checkfixing(circle c1, circle c2)
{
  inversion iota = inversion(c1, c2, 0);
  circle img1 = iota * c1;
  circle img2 = iota * c2;
  assert(close(locate(img1.C), locate(c1.C)) && nearly(img1.r, c1.r));
  assert(close(locate(img2.C), locate(c2.C)) && nearly(img2.r, c2.r));
  assert(close(locate(iota.C), locate(radicalcenter(c1, c2))));
}

void run()
{
  point P, Q;

  // one circle strictly contains the other (the example from issue #143):
  // both inversions exist and both must map the first circle to the second
  P = (0, 0); Q = (0.7, 1);
  circle c1 = circle(P, 3), c2 = circle(Q, 0.7);
  checkinversion(c1, c2, 1);
  checkinversion(c1, c2, -1);

  // two circles outside each other
  P = (-2, 0); Q = (2, 0);
  c1 = circle(P, 1); c2 = circle(Q, 1.5);
  checkinversion(c1, c2, 1);
  checkinversion(c1, c2, -1);
  checkfixing(c1, c2);

  // two circles intersecting at two points
  P = (0, 0); Q = (1.5, 0);
  c1 = circle(P, 2); c2 = circle(Q, 1.5);
  checkinversion(c1, c2, 1);
  checkinversion(c1, c2, -1);
  checkfixing(c1, c2);

  // externally tangent: sgn < 0 inversion exists; sgn > 0 falls back to it
  P = (0, 0); Q = (3, 0);
  c1 = circle(P, 1); c2 = circle(Q, 2);
  checkinversion(c1, c2, -1);
  checkinversion(c1, c2, 1, true);   // falls back to the sgn < 0 inversion

  // internally tangent: sgn > 0 inversion exists; sgn < 0 falls back to it
  P = (0, 0); Q = (2, 0);
  c1 = circle(P, 3); c2 = circle(Q, 1);
  checkinversion(c1, c2, 1);
  checkinversion(c1, c2, -1, true);  // falls back to the sgn > 0 inversion

  // concentric: both inversions exist, centered at the common center
  P = (1, 1); Q = (1, 1);
  c1 = circle(P, 2); c2 = circle(Q, 5);
  checkinversion(c1, c2, 1);
  checkinversion(c1, c2, -1);

  // equal radii: only one inversion exists (center at the midpoint);
  // sgn < 0 falls back to it (used to be a divide-by-zero crash)
  P = (0, 0); Q = (3, 0);
  c1 = circle(P, 1); c2 = circle(Q, 1);
  checkinversion(c1, c2, 1);
  checkinversion(c1, c2, -1, true);  // falls back to the sgn > 0 inversion
}

run();

currentcoordsys = cartesiansystem(i = N, j = W);
run();

currentcoordsys = cartesiansystem(i = dir(30), j = dir(120));
run();

EndTest();

