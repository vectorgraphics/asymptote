import TestLib;
import graph3;

// Tests for surface.buildNeighborsSlow, which reconstructs the adjacency
// table `neighbors` from the patch geometry alone (without consulting the
// u/v grid in `index`).

real tol = 1e-6;
bool near(triple x, triple y) { return abs(x - y) < tol; }

path3 patchboundary(patch p) {
  return p.triangular ? p.externaltriangular() : p.external();
}

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
      path3 extp = patchboundary(s.s[p]);
      path3 extq = patchboundary(s.s[ei.patch]);
      triple a1 = point(extp, e), a2 = point(extp, e + 1);
      triple b1 = point(extq, ei.edge), b2 = point(extq, ei.edge + 1);
      if (ei.reversed) { triple t = b1; b1 = b2; b2 = t; }
      assert(near(a1, b1) && near(a2, b2),
             "adjacent patch edges must coincide geometrically");
    }
  }
}

// Collect the boundary edges (those with no adjoining patch) as endpoint
// pairs.
triple[][] boundaryEdges(surface s) {
  triple[][] edges;
  for (int k = 0; k < s.neighbors.length; ++k) {
    path3 ext = patchboundary(s.s[k]);
    for (int e = 0; e < s.neighbors[k].length; ++e)
      if (s.neighbors[k][e].patch < 0)
        edges.push(new triple[] {point(ext, e), point(ext, e + 1)});
  }
  return edges;
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

  // The grid constructor records v-closure but leaves the u-seam open.
  assert(boundaryEdges(s).length > 0,
         "grid construction should leave the u-seam open");

  s.buildNeighborsSlow();
  checkreciprocal(s);
  assert(boundaryEdges(s).length == 0,
         "buildNeighborsSlow must close the Klein bottle (no boundary edges)");
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
  // It is the only adjacency: every other edge is a boundary.
  assert(boundaryEdges(s).length == 6,
         "only the loop edge is shared (4+4 edges, 2 matched)");
}
EndTest();

StartTest("unithemisphere has the expected equatorial boundary");
{
  surface s = surface(unithemisphere);
  s.buildNeighborsSlow();
  checkreciprocal(s);

  // The only boundary is the equator z = 0: four quarter-arc edges joining
  // (1,0,0), (0,1,0), (-1,0,0), (0,-1,0) into a single closed loop.
  triple[][] edges = boundaryEdges(s);
  assert(edges.length == 4, "hemisphere boundary must be four edges");

  triple[] rim = {(1,0,0), (0,1,0), (-1,0,0), (0,-1,0)};
  for (triple[] edge : edges) {
    for (triple end : edge) {
      assert(abs(end.z) < tol, "boundary lies on the equator z = 0");
      assert(abs(abs(end) - 1) < tol, "boundary lies on the unit circle");
    }
  }

  // Each rim vertex must be met by exactly two boundary edges (one
  // incoming, one outgoing): the boundary is a single closed cycle.
  for (triple v : rim) {
    int starts = 0, ends = 0;
    for (triple[] edge : edges) {
      if (near(edge[0], v)) ++starts;
      if (near(edge[1], v)) ++ends;
    }
    assert(starts == 1 && ends == 1,
           "each equator vertex joins two boundary edges");
  }
}
EndTest();
