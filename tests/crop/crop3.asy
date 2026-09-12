import TestLib;
import crop3;
StartTest("crop3");

// The furthest any point of a cropped surface strays outside the region
// {f >= 0} it was cropped to, as a distance: -f/|grad f| puts fields of
// different units -- a plane's signed distance, a ball's difference of squares
// -- on the same scale.
real leak(patch[] c, real f(triple), triple grad(triple) = nGrad(f)) {
  real worst = 0;
  for (patch q : c) {
    for (int i = 0; i <= 8; ++i) {
      for (int j = 0; j <= 8; ++j) {
        real u = i/8, v = j/8;
        if (q.triangular && u + v > 1) continue;
        triple x = q.point(u, v);
        real g = abs(grad(x));
        if (g > 0) worst = max(worst, -f(x) / g);
      }
    }
  }
  return worst;
}

// How nearly parallel, over a cropped surface, each patch normal is to the
// direction the surface faced before cropping. Only the magnitude is checked:
// patchwithnormals may return a patch whose parametrization runs the other way,
// so an antiparallel patch.normal() is permitted (see smoothcontour3.asy). A
// patch that is degenerate or lies askew to the surface sends this below one.
real worstalignment(patch[] c, triple facing(triple)) {
  real worst = realMax;
  for (patch q : c) {
    pair ctr = q.triangular ? (1/3, 1/3) : (0.5, 0.5);
    triple n = surfacenormal(q, ctr);
    worst = min(worst, abs(dot(unit(n), unit(facing(q.point(ctr.x, ctr.y))))));
  }
  return worst;
}

{
  // halfspace reads the bound in the same scale as the normal: the region is
  // dot(p, normal) >= value, so scaling the normal without scaling the bound
  // moves the plane.
  cropfield unit = halfspace(Z, '>=', -1);
  cropfield scaled = halfspace((0,0,2), '>=', -2);
  cropfield mismatched = halfspace((0,0,2), '>=', -1);
  assert(close(unit.f((0,0,-1)), 0));
  assert(close(scaled.f((0,0,-1)), 0));
  assert(close(mismatched.f((0,0,-0.5)), 0));
  assert(!close(mismatched.f((0,0,-1)), 0));
}

{
  // A plane cut of the unit sphere. The cut is tilted off the zero set by the
  // order of crop3_settings.regularizer, so it leaks by about 7e-5.
  real f(triple p) { return p.z + 0.3; }
  patch[] c = crop(unitsphere, f, new triple(triple p) { return Z; }).s;
  assert(c.length > 0);
  assert(leak(c, f) < 1e-3);
}

{
  // Near tangency: a plane grazing the top of the sphere still cuts cleanly.
  for (real eps : new real[] {1e-1, 1e-2, 1e-3}) {
    real f(triple p) { return p.z - (1 - eps); }
    patch[] c = crop(unitsphere, f, new triple(triple p) { return Z; }).s;
    assert(c.length > 0);
    assert(leak(c, f) < 1e-3);
  }
}

{
  // A unit square with the corner bitten out by a disk about it. Past a radius
  // of about 0.7 the average of the corners lies outside the region it would
  // have to fan, so fancentre has to search for a centre that works.
  patch sq = patch((0,0,0)--(1,0,0)--(1,1,0)--(0,1,0)--cycle);
  triple up(triple) { return Z; }
  for (real r : new real[] {0.5, 0.7, 0.8, 0.9}) {
    cropfield outside = ball((0,0,0), r, '>=');
    patch[] c = crop(sq, outside);
    assert(c.length > 0);
    assert(worstalignment(c, up) > 0.99);
    assert(leak(c, outside.f, outside.grad) < 1e-3);
  }
}

{
  // Cropping to a box is six half-space crops in turn, not one crop to the
  // minimum of the six fields; the result must stay inside the box.
  triple lo = (-0.6,-0.6,-0.6), hi = (0.6,0.6,0.6);
  patch[] c = crop(unitsphere, boxsides(lo, hi)).s;
  assert(c.length > 0);
  for (cropfield side : boxsides(lo, hi))
    assert(leak(c, side.f, side.grad) < 1e-3);
}

{
  // A patch wholly inside the region is kept whole; one wholly outside is
  // dropped.
  patch sq = patch((0,0,0)--(1,0,0)--(1,1,0)--(0,1,0)--cycle);
  assert(crop(sq, halfspace(Z, '>=', -1)).length == 1);
  assert(crop(sq, halfspace(Z, '>=', 1)).length == 0);
}

// Patches lifted from examples/teapot.asy, which crops much harder than
// anything analytic: strong curvature, high aspect ratios, and corners sitting
// exactly on the planes one naturally crops by.
triple[][] spoutbody = {
  {(48.18897,0,40.3937),(73.70078,0,40.3937),(65.19685,0,59.52755),(76.53543,0,68.0315)},
  {(48.18897,-18.70866,40.3937),(73.70078,-18.70866,40.3937),(65.19685,-7.086619,59.52755),(76.53543,-7.086619,68.0315)},
  {(48.18897,-18.70866,17.00787),(87.87401,-18.70866,23.38582),(68.0315,-7.086619,57.40157),(93.5433,-7.086619,68.0315)},
  {(48.18897,0,17.00787),(87.87401,0,23.38582),(68.0315,0,57.40157),(93.5433,0,68.0315)}
};
triple[][] spoutlip = {
  {(76.53543,0,68.0315),(79.37007,0,70.15748),(82.20472,0,70.15748),(79.37007,0,68.0315)},
  {(76.53543,-7.086619,68.0315),(79.37007,-7.086619,70.15748),(82.20472,-4.251961,70.15748),(79.37007,-4.251961,68.0315)},
  {(93.5433,-7.086619,68.0315),(99.92125,-7.086619,70.68897),(97.79527,-4.251961,71.22047),(90.70866,-4.251961,68.0315)},
  {(93.5433,0,68.0315),(99.92125,0,70.68897),(97.79527,0,71.22047),(90.70866,0,68.0315)}
};
// A quarter of the lid: much flatter than the spout, which is what makes it
// awkward. See the oblique-plane case below.
triple[][] lidnext = {
  {(-5.669294,0,76.53543),(-11.33858,0,72.28346),(-36.85039,0,72.28346),(-36.85039,0,68.0315)},
  {(-5.669294,3.174809,76.53543),(-11.33858,6.349609,72.28346),(-36.85039,20.63622,72.28346),(-36.85039,20.63622,68.0315)},
  {(-3.174809,5.669294,76.53543),(-6.349609,11.33858,72.28346),(-20.63622,36.85039,72.28346),(-20.63622,36.85039,68.0315)},
  {(0,5.669294,76.53543),(0,11.33858,72.28346),(0,36.85039,72.28346),(0,36.85039,68.0315)}
};

{
  // The spout lip cut by a ball centred beyond its tip. The kept region has
  // five sides, so it is fanned, on a patch curving hard in both directions.
  cropfield c = ball((93,0,68), 15, '<=');
  patch[] out = crop(patch(spoutlip), c);
  assert(out.length == 5);
  assert(leak(out, c.f, c.grad) < 0.05);
}

{
  // The lid cut by a plane oblique to everything: another five-sided region,
  // but a nearly flat one, so the three edge normals of a fan triangle come out
  // nearly parallel. That leaves trianglewithnormals free to satisfy them with
  // an inner control point far outside the triangle, which used to bulge one
  // fan triangle 3.2 units through the plane while its whole boundary stayed
  // on the correct side.
  real f(triple p) { return (p.x + p.y + p.z)/sqrt(3) - 40; }
  triple grad(triple) { return unit((1,1,1)); }
  patch[] out = crop(patch(lidnext), f, grad);
  assert(out.length == 5);
  assert(leak(out, f, grad) < 0.05);
}

{
  // The spout body cut by a plane through it. 40.3937 and 68.0315 are exact
  // corner z coordinates, so crossings land on corners and are absorbed by
  // snaptolerance instead of making slivers.
  for (real z : new real[] {20, 30, 40, 40.3937, 55, 60}) {
    real f(triple p) { return p.z - z; }
    patch[] out = crop(patch(spoutbody), f, new triple(triple) { return Z; });
    assert(out.length > 0);
    assert(leak(out, f) < 0.05);
  }
  // Both kept corners lie exactly on this plane, so the region that survives is
  // the top edge alone: zero area, and nothing to draw.
  real top(triple p) { return p.z - 68.0315; }
  assert(crop(patch(spoutbody), top,
              new triple(triple) { return Z; }).length == 0);
}

{
  // Cropping is an approximation: one cubic per region side cannot follow a
  // strongly curved cut across a big patch. The error must fall when the patch
  // is subdivided first -- here from about a third of a unit to a thirtieth,
  // on a teapot some 185 units across.
  cropfield c = ball((48.18897,0,17.00787), 45, '<=');
  real coarse = leak(crop(patch(spoutbody), c), c.f, c.grad);
  real fine = 0;
  for (triple[][] half : hsplit(spoutbody, 0.5))
    for (triple[][] quarter : vsplit(half, 0.5))
      fine = max(fine, leak(crop(patch(quarter), c), c.f, c.grad));
  assert(coarse > 0.1);        // the coarse cut really does stray
  assert(fine < coarse / 4);
}

{
  // A patch with two coincident corners: patch.normal returns zero there, and
  // surfacenormal has to nudge inwards to recover a direction.
  triple[][] pinched = {
    {(0,0,0),(0,0,0),(0,0,0),(0,0,0)},
    {(1,0,0),(1,0.3,0),(1,0.7,0),(1,1,0)},
    {(2,0,0.5),(2,0.3,0.5),(2,0.7,0.5),(2,1,0.5)},
    {(3,0,0),(3,0.3,0),(3,0.7,0),(3,1,0)}
  };
  real f(triple p) { return p.x - 1.5; }
  patch[] out = crop(patch(pinched), f, new triple(triple) { return X; });
  assert(out.length > 0);
  assert(leak(out, f) < 0.05);
}

EndTest();
