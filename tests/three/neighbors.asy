import TestLib;
import graph3;

// Tests for the edge information a surface carries in `neighbors`: for
// surface.buildNeighborsSlow, which reconstructs that table from the patch
// geometry alone (without consulting the u/v grid in `index`), and for
// surface.boundary, which reads it back to assemble the boundary loops.

real tol = 1e-6;
bool near(triple x, triple y) { return abs(x - y) < tol; }

// Every recorded adjacency must be reciprocal and the shared edge must
// coincide geometrically (respecting the reversed flag).
void checkreciprocal(surface s) {
  edgeInfo[][] nbrs = s.neighbors;
  assert(nbrs.length == s.s.length, "one neighbor row per patch");
  for (int p = 0; p < nbrs.length; ++p) {
    for (int e = 0; e < nbrs[p].length; ++e) {
      edgeInfo ei = nbrs[p][e];
      if (ei.patch < 0) continue;
      edgeInfo back = nbrs[ei.patch][ei.edge];
      assert(back.patch == p && back.edge == e && back.reversed == ei.reversed,
             "adjacency must be reciprocal");
      path3 extp = s.s[p].external();
      path3 extq = s.s[ei.patch].external();
      triple a1 = point(extp, e), a2 = point(extp, e + 1);
      triple b1 = point(extq, ei.edge), b2 = point(extq, ei.edge + 1);
      if (ei.reversed) { triple t = b1; b1 = b2; b2 = t; }
      assert(near(a1, b1) && near(a2, b2),
             "adjacent patch edges must coincide geometrically");
    }
  }
}

// The number of patch edges that adjoin another patch.
int sharedEdges(surface s) {
  int shared = 0;
  for (int k = 0; k < s.neighbors.length; ++k)
    for (int e = 0; e < s.neighbors[k].length; ++e)
      if (s.neighbors[k][e].patch >= 0) ++shared;
  return shared;
}

// How many times a closed loop winds about the z axis, signed according to
// the direction of travel.  The loop must avoid the axis.
real windings(path3 g, int samplesPerArc = 8) {
  real turning = 0;
  for (int i = 0; i < samplesPerArc*length(g); ++i) {
    triple p = point(g, i/samplesPerArc), q = point(g, (i + 1)/samplesPerArc);
    turning += degrees(angle((q.x, q.y)/(p.x, p.y)));  // complex division
  }
  return turning/360;
}

// Assert that a loop is the circle of the given radius in the plane
// z = height, traversed exactly once.
void checkcircle(path3 g, real radius, real height, real circletol = 1e-3) {
  assert(cyclic(g), "a boundary loop must be cyclic");
  int samples = 16;
  for (int i = 0; i < samples*length(g); ++i) {
    triple p = point(g, i/samples);
    assert(abs(p.z - height) < circletol, "the loop lies in its plane");
    assert(abs(abs((p.x, p.y)) - radius) < circletol,
           "the loop lies on its circle");
  }
  assert(abs(abs(windings(g)) - 1) < 1e-2,
         "the loop winds once about the axis");
}

StartTest("Klein bottle is a closed surface");
{
  // The standard figure-eight Klein bottle.  Its parametrization is not
  // u-periodic in the bookkeeping sense (the u=0 and u=2pi circles are
  // traced in opposite directions), so the grid constructor leaves the
  // u-seam as a boundary; but the two circles coincide as point sets, so
  // buildNeighborsSlow recovers the full closure.
  //
  // [Side note: A non-orientable surface cannot be both u-periodic and
  // v-periodic in the bookkeeping sense, since that same bookkeeping would
  // produce an orientation.]
  //
  triple f(pair t) {
    real u = t.x, v = t.y;
    real r = 2 - cos(u);
    real x = 3*cos(u)*(1 + sin(u)) + r*cos(v)*(u < pi ? cos(u) : -1);
    real y = 8*sin(u) + (u < pi ? r*sin(u)*cos(v) : 0);
    real z = r*sin(v);
    return (x, y, z);
  }
  surface s = surface(f, (0,0), (2pi,2pi), 8, 8, Spline);

  // The grid constructor records v-closure but leaves the u-seam open, as
  // the two circles it is made of.
  assert(s.boundary().length == 2,
         "grid construction should leave the u-seam open");

  s.buildNeighborsSlow();
  checkreciprocal(s);
  assert(s.boundary().length == 0,
         "buildNeighborsSlow must close the Klein bottle (no boundary)");
}
EndTest();

StartTest("buildNeighborsSlow agrees with constructed neighbors on a torus");
{
  // A torus is fully closed and its grid constructor populates every
  // adjacency, so the geometry-only reconstruction must reproduce it
  // exactly (same partner, same edge, same orientation).
  real R = 3, r = 1;
  triple torus(pair t) {
    real u = t.x, v = t.y;
    return ((R + r*cos(v))*cos(u), (R + r*cos(v))*sin(u), r*sin(v));
  }
  surface s = surface(torus, (0,0), (2pi,2pi), 8, 10, Spline);

  edgeInfo[][] constructed = copyNeighbors(s.neighbors);
  assert(constructed.length == s.s.length, "constructed neighbors present");

  s.buildNeighborsSlow();
  assert(s.neighbors.length == constructed.length, "row count preserved");
  for (int k = 0; k < constructed.length; ++k) {
    assert(s.neighbors[k].length == constructed[k].length,
           "edge count preserved");
    for (int e = 0; e < constructed[k].length; ++e) {
      edgeInfo a = s.neighbors[k][e], b = constructed[k][e];
      assert(a.patch == b.patch && a.edge == b.edge && a.reversed == b.reversed,
             "reconstructed adjacency must match the constructed one");
    }
  }
}
EndTest();

StartTest("patches sharing a nondegenerate loop edge are adjacent");
{
  // Two patches joined along a single shared edge that is a loop: its first
  // and last control points coincide at the origin, but its interior
  // control points (1,+-1,0) bulge out to +x, so the edge is nondegenerate
  // even though its endpoints are equal.  Patch A lies in the z = 0 plane
  // and patch B in the x = 0 plane; they meet only along the loop, which
  // they trace in opposite directions.
  triple[][] A = {
    {(-2,1,0),       (-2,0.33,0),  (-2,-0.33,0), (-2,-1,0)},
    {(-1.33,0.66,0), (-1,0.4,0),   (-1,-0.4,0),  (-1.33,-0.66,0)},
    {(-0.66,0.33,0), (-0.5,0.2,0), (-0.5,-0.2,0),(-0.66,-0.33,0)},
    {(0,0,0),        (1,1,0),      (1,-1,0),     (0,0,0)}  // edge 1: the loop
  };
  triple[][] B = {
    {(0,1,2),        (0,0.33,2),   (0,-0.33,2),  (0,-1,2)},
    {(0,0.66,1.33),  (0,0.5,1),    (0,-0.5,1),   (0,-0.66,1.33)},
    {(0,0.33,0.66),  (0,0.25,0.5), (0,-0.25,0.5),(0,-0.33,0.66)},
    {(0,0,0),        (1,-1,0),     (1,1,0),      (0,0,0)} // edge 1: loop rev'd
  };
  surface s = surface(patch(A), patch(B));

  // The loop edge is edge 1 of each patch; its endpoints coincide but it is
  // nondegenerate, so it must be matched (not discarded as degenerate).
  assert(near(point(s.s[0].external(), 1), point(s.s[0].external(), 2)),
         "the shared edge really is a loop (equal endpoints)");

  s.buildNeighborsSlow();
  checkreciprocal(s);
  edgeInfo ei = s.neighbors[0][1];
  assert(ei.patch == 1 && ei.edge == 1 && ei.reversed,
         "the loop edge must join patch 0 to patch 1 (reversed)");
  assert(sharedEdges(s) == 2,
         "the loop edge is the only one shared (of 4+4 edges)");

  // Each patch leaves three consecutive edges unshared, so assembling the
  // boundary steps from one edge of a patch to the next without crossing a
  // seam -- the case in which the rotation about a corner stops at once.
  // (Every other surface here is grid- or ring-like, leaving a patch's
  // unshared edges on opposite sides, so the rotation always has to cross.)
  path3[] b = s.boundary();
  assert(b.length == 1, "the pinched pair has a single boundary loop");
  path3 g = b[0];
  assert(cyclic(g) && length(g) == 6,
         "the loop uses all three free edges of each patch");
  assert(near(point(g, 1), O) && near(point(g, 4), O),
         "the loop passes through the pinch point twice");
}
EndTest();

StartTest("unithemisphere has the expected equatorial boundary");
{
  surface s = surface(unithemisphere);
  s.buildNeighborsSlow();
  checkreciprocal(s);

  // The only boundary is the equator: a smooth closed loop traversing the
  // unit circle in the plane z = 0 exactly once.  How the hemisphere is cut
  // into patches -- and hence how many arcs that loop is made of -- is an
  // implementation detail, so nothing below depends on it.
  path3[] b = s.boundary();
  assert(b.length == 1, "the hemisphere boundary is a single loop");
  path3 g = b[0];
  checkcircle(g, 1, 0);

  // The loop is smooth: consecutive arcs leave and arrive in the same
  // direction (a distance of 0.02 between unit tangents is about 1 degree).
  for (int i = 0; i < length(g); ++i)
    assert(abs(dir(g, i, 1) - dir(g, i, -1)) < 0.02,
           "the boundary loop must be smooth (no corners at the joins)");

  // Patches meeting the equator in a single corner, with no edge along it,
  // lie between the patches that do contribute an arc; assembling the loop
  // has to pass through them.
  int passthrough = 0;
  for (int k = 0; k < s.s.length; ++k) {
    path3 ext = s.s[k].external();
    bool corner = false, arc = false;
    for (int e = 0; e < s.neighbors[k].length; ++e) {
      if (abs(point(ext, e).z) < tol) corner = true;
      if (s.neighbors[k][e].patch < 0) arc = true;
    }
    if (corner && !arc) ++passthrough;
  }
  assert(passthrough > 0,
         "some patch must meet the equator in a single corner only");
}
EndTest();

StartTest("a Mobius band is bounded by one loop winding twice");
{
  // A Mobius band one patch wide.  The parametrization rotates the cross
  // section at u by u/2, spreading the twist along the whole band; by the
  // time the cross section comes back round it has flipped, so that
  // f(2pi,v) = f(0,-v).  The band is therefore nonorientable without being
  // u-periodic: the grid constructor leaves the u = 0 seam open, and
  // buildNeighborsSlow has to recover it from the geometry.
  int n = 12;
  real R = 2, w = 0.6;
  triple mobius(pair z) {
    real u = z.x, v = z.y;
    real r = R + v*cos(u/2);
    return (r*cos(u), r*sin(u), v*sin(u/2));
  }
  surface s = surface(mobius, (0,-w), (2pi,w), n, 1, Spline);
  assert(s.s.length == n, "the band is n patches around and one wide");
  assert(!s.ucyclic(), "the band cannot be recorded as u-periodic");

  // Every cross section is a straight segment, so the two patches meeting
  // along one share it exactly rather than to within a spline fuzz.
  for (int k = 0; k < s.s.length; ++k) {
    path3 ext = s.s[k].external();
    for (int e : new int[] {1, 3}) {
      triple a = point(ext, e), b = point(ext, e + 1);
      assert(near(postcontrol(ext, e), interp(a, b, 1/3)) &&
             near(precontrol(ext, e + 1), interp(a, b, 2/3)),
             "the cross sections are straight segments");
    }
  }

  s.buildNeighborsSlow();
  checkreciprocal(s);

  // Edge 1 of a patch is the cross section it shares with the next one.  At
  // the seam the half twist leaves the two patches running along that cross
  // section in the same direction rather than in opposite directions.
  edgeInfo seam = s.neighbors[n-1][1];
  assert(seam.patch == 0 && seam.edge == 3 && !seam.reversed,
         "the half twist makes the patches at the seam agree in direction");
  assert(sharedEdges(s) == 2*n, "every cross section is shared");

  // Each patch leaves its two lengthwise edges unshared, one along each
  // side of the band, and all 2n of them belong to a single loop: the band
  // has one boundary curve, which runs the length of one side and then the
  // length of the other.
  path3[] b = s.boundary();
  assert(b.length == 1, "the Mobius band has a single boundary loop");
  path3 g = b[0];
  assert(cyclic(g), "a boundary loop must be cyclic");
  assert(length(g) == 2*n, "the loop uses every unshared edge");
  assert(abs(abs(windings(g)) - 2) < 1e-2,
         "the boundary must wind twice about the axis");
}
EndTest();

StartTest("a cube with a face missing is bounded by that face's outline");
{
  triple[] v = {(0,0,0), (1,0,0), (1,1,0), (0,1,0),
                (0,0,1), (1,0,1), (1,1,1), (0,1,1)};
  patch face(int a, int b, int c, int d) {
    return patch(v[a]--v[b]--v[c]--v[d]--cycle);
  }
  // The five faces other than z = 1, each wound so its normal points out.
  surface s = surface(face(0,3,2,1), face(0,1,5,4), face(1,2,6,5),
                      face(2,3,7,6), face(3,0,4,7));
  s.buildNeighborsSlow();
  checkreciprocal(s);

  // Each of the four side faces contributes its one top edge, and together
  // they outline the missing face.
  path3[] b = s.boundary();
  assert(b.length == 1, "the missing face leaves a single boundary loop");
  path3 g = b[0];
  assert(cyclic(g) && length(g) == 4, "the loop is a quadrilateral");
  assert(piecewisestraight(g), "its sides are straight");

  // Its corners are the four corners of the missing face, each visited
  // once, and consecutive ones are joined along an edge of that face.
  triple[] corners = {v[4], v[5], v[6], v[7]};
  bool[] seen = array(4, false);
  for (int i = 0; i < 4; ++i) {
    int at = -1;
    for (int j = 0; j < 4; ++j)
      if (near(point(g, i), corners[j])) at = j;
    assert(at >= 0, "every corner of the loop is a corner of the face");
    assert(!seen[at], "no corner is visited twice");
    seen[at] = true;
    assert(abs(abs(point(g, i + 1) - point(g, i)) - 1) < tol,
           "consecutive corners are joined by an edge of the face");
  }
}
EndTest();

StartTest("a cylinder is bounded by its two rim circles");
{
  surface s = surface(unitcylinder);
  s.buildNeighborsSlow();
  checkreciprocal(s);

  path3[] b = s.boundary();
  assert(b.length == 2, "the cylinder has two boundary loops");
  real z0 = point(b[0], 0).z, z1 = point(b[1], 0).z;
  checkcircle(b[0], 1, z0);
  checkcircle(b[1], 1, z1);
  assert(abs(min(z0, z1)) < tol && abs(max(z0, z1) - 1) < tol,
         "the loops are the circles z = 0 and z = 1");
}
EndTest();

StartTest("a cylinder two patches tall, cut in half, is bounded by four");
{
  // Two rings of patches stacked to make a cylinder two patches tall.
  surface s;
  s.s.append((zscale3(0.5)*unitcylinder).s);
  int split = s.s.length;
  s.s.append((shift(0.5Z)*zscale3(0.5)*unitcylinder).s);
  s.buildNeighborsSlow();
  checkreciprocal(s);
  assert(s.boundary().length == 2,
         "joined, the two rings still have only the two rim circles");

  // Forget that the rings adjoin.  The circle at z = 1/2 is then unshared
  // from both sides, so it bounds each ring separately.
  for (int k = 0; k < s.s.length; ++k)
    for (int e = 0; e < s.neighbors[k].length; ++e) {
      edgeInfo ei = s.neighbors[k][e];
      if (ei.patch >= 0 && (ei.patch < split) != (k < split))
        s.neighbors[k][e] = new edgeInfo;
    }
  checkreciprocal(s);

  path3[] b = s.boundary();
  assert(b.length == 4, "the two rings have four boundary loops together");
  int[] count = array(3, 0);  // loops at z = 0, 1/2 and 1
  for (path3 g : b) {
    real z = point(g, 0).z;
    checkcircle(g, 1, z);
    int i = round(2*z);
    assert(i >= 0 && i <= 2 && abs(z - 0.5*i) < tol,
           "every loop is one of the circles z = 0, 1/2, 1");
    ++count[i];
  }
  assert(count[0] == 1 && count[2] == 1, "one loop at each rim");
  assert(count[1] == 2, "two coincident loops along the cut");
}
EndTest();
