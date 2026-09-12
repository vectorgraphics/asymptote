import three;
import math;
from smoothcontour3 access patchwithnormals, trianglewithnormals, nGrad;

struct crop3_settings {
  // A crossing this close, in boundary time, to a corner is snapped onto it.
  // Shorter slivers defeat the bisection and leave the withnormals routines
  // ill conditioned.
  static real snaptolerance = 1e-4;

  // How much of the direction orthogonal to the patch boundary is mixed into a
  // cut; see cutdirection. It buys sane behaviour near tangency at the cost of
  // tilting every other cut off the zero set by the same order.
  static real regularizer = 1e-3;

  // Fan every cropped region, even one with few enough sides for one patch.
  static bool alwaysfan = false;
}

/***********************************************/
/************ THE PARAMETER DOMAIN *************/
/***********************************************/
// Every point of interest is tracked by its coordinates in the parameter domain
// of the patch: the unit square, or the triangle {u >= 0, v >= 0, u + v <= 1}.
// Both are convex, so a point constructed between others stays in the domain,
// where positions, normals and colours are all exact. New edges are laid out
// here, then pushed forward onto the patch.

// The corners of the parameter domain, in the order patch.external() visits
// them: counterclockwise, so the domain lies to the left of each edge.
private pair[] cornerdomain(patch s) {
  return s.triangular ? new pair[] {(0,0), (1,0), (0,1)}
                      : new pair[] {(0,0), (1,0), (1,1), (0,1)};
}

// Laying out a new edge measures angles and lengths, which the native triangle
// parameters distort: its right angle is an artefact, so a cut would depend on
// which corner Asymptote singles out. Hence equilateral coordinates here;
// everything invariant stays in the native ones.
private pair regular(patch s, pair uv) {
  return s.triangular ? (uv.x + 0.5*uv.y, 0.5*sqrt(3)*uv.y) : uv;
}

// Its inverse, for feeding a constructed point back to patch.point/normal.
private pair native(patch s, pair x) {
  return s.triangular ? (x.x - x.y/sqrt(3), 2*x.y/sqrt(3)) : x;
}

// The columns of the Jacobian, in the regularized coordinates.
private triple[] frame(patch s, pair uv) {
  if (s.triangular) {
    // The triangular partials are a third of the directional derivatives, and
    // the regularized directions pull back to (1,0) and (-1,2)/sqrt(3).
    triple su = 3 * s.partialutriangular(uv.x, uv.y);
    triple sv = 3 * s.partialvtriangular(uv.x, uv.y);
    return new triple[] {su, (2*sv - su) / sqrt(3)};
  }
  return new triple[] {s.partialu(uv.x, uv.y), s.partialv(uv.x, uv.y)};
}

private triple pushforward(triple[] J, pair w) {
  return w.x*J[0] + w.y*J[1];
}
// not a pseudoinverse of pushforward
private pair pullback(triple[] J, triple g) {
  return (dot(J[0], g), dot(J[1], g));
}

// The unit normal of edge k of the regularized domain, pointing inwards.
private pair inwardnormal(patch s, pair[] cuv, int k) {
  pair d = regular(s, cuv[(k+1) % cuv.length]) - regular(s, cuv[k]);
  return unit((-d.y, d.x));
}

// The domain coordinates at time t of the patch boundary: affine in t along
// each segment, which is the patch restricted to one domain edge.
private pair uvattime(pair[] cuv, real t) {
  int n = cuv.length;
  int i = floor(t);
  real frac = t - i;
  i = i % n;
  if (i < 0) i += n;
  pair p0 = cuv[i], p1 = cuv[(i+1) % n];
  return p0 + frac * (p1 - p0);
}

// patch.normal legitimately returns zero where the parametrization degenerates;
// nudging towards the centre of the domain recovers it.
triple surfacenormal(patch s, pair uv) {
  triple n = s.normal(uv.x, uv.y);
  if (abs(n) != 0) return n;
  pair centre = s.triangular ? (1/3, 1/3) : (0.5, 0.5);
  for (real eps = 1e-4; eps < 1; eps *= 10) {
    pair w = uv + eps * (centre - uv);
    n = s.normal(w.x, w.y);
    if (abs(n) != 0) return n;
  }
  return n;
}

private pen colorat(patch s, pair uv) {
  pen[] c = s.colors;
  if (s.triangular) {
    real v = min(max(uv.y, 0), 1);
    if (v == 1) return c[2];
    real u = min(max(uv.x / (1 - v), 0), 1);
    return interp(interp(c[0], c[1], u), c[2], v);
  }
  return interp(interp(c[0], c[1], uv.x), interp(c[3], c[2], uv.x), uv.y);
}

/***********************************************/
/*********** REGIONS OF A CROPPED PATCH ********/
/***********************************************/

// A vertex of a cropped region's boundary. One taken from the patch boundary
// keeps the position computed there, so curves meeting at it agree exactly.
private struct cropvertex {
  pair uv;          // parameter-domain coordinates
  triple position;  // the corresponding point of the patch
  void operator init(pair uv, triple position) {
    this.uv = uv;
    this.position = position;
  }
  void operator init(patch s, pair uv) {
    operator init(uv, s.point(uv.x, uv.y));
  }
}

// One boundary segment of a cropped region. The endpoints supply the normals
// the withnormals routines need; the domain preimage lets a fan test its centre
// with ordinary two-dimensional path predicates.
private struct cropedge {
  path3 g;
  path domain;
  pair uv0, uv1;
  void operator init(path3 g, path domain, pair uv0, pair uv1) {
    this.g = g;
    this.domain = domain;
    this.uv0 = uv0;
    this.uv1 = uv1;
  }
}

private cropedge reverse(cropedge e) {
  return cropedge(reverse(e.g), reverse(e.domain), e.uv1, e.uv0);
}

private triple normalat(patch s, cropedge e, real t) {
  return surfacenormal(s, native(s, point(e.domain, t)));
}

// The time in [k, k+1] at which segment k of the patch boundary e crosses the
// zero set of f. Keeping it a time, never a point, is what lets two patches
// sharing an edge find the same crossing and keep an exact subpath of it.
private real edgecrossing(path3 e, int k, real fa, real fb,
                          real f(triple), triple grad(triple)) {
  real g(real t) { return f(point(e, t)); }
  real t = findroot(g, k, k+1, fa=fa, fb=fb);

  // One Newton step.
  triple velocity = bezierP(point(e, k), postcontrol(e, k),
                            precontrol(e, k+1), point(e, k+1), t - k);
  triple p = point(e, t);
  real slope = dot(grad(p), velocity);
  if (slope != 0) {
    real refined = t - f(p) / slope;
    if (k <= refined && refined <= k+1) t = refined;
  }

  // Absorb a crossing that is indistinguishable from a corner.
  real tol = crop3_settings.snaptolerance;
  if (t - k < tol) t = k;
  else if (k+1 - t < tol) t = k+1;
  return t;
}

// The cubic meeting the patch at both ends, with the end velocities of a path
// laid out in the regularized domain pushed forward onto it.
private cropedge lift(patch s, cropvertex v1, cropvertex v2, path domain) {
  triple c1 = v1.position + pushforward(frame(s, v1.uv),
                                        postcontrol(domain, 0)
                                        - point(domain, 0));
  triple c2 = v2.position - pushforward(frame(s, v2.uv),
                                        point(domain, 1)
                                        - precontrol(domain, 1));
  return cropedge(v1.position .. controls c1 and c2 .. v2.position,
                  domain, v1.uv, v2.uv);
}

// The direction, in the regularized domain, in which a cut leaves or arrives at
// a crossing. A cut follows the zero set of g = f o S, so its tangent is
// grad g = transpose(Jacobian) grad f turned through a right angle, signed by
// the traversal orientation to keep {g >= 0} on the left. Where the zero set is
// tangent to the patch grad g vanishes; the edge's inward normal is the
// fallback, so that patches meeting smoothly across the edge cut smoothly too.
private pair cutdirection(patch s, pair[] cuv, cropvertex v, int edge,
                          bool inward, triple gradf) {
  triple[] J = frame(s, v.uv);
  pair g = pullback(J, gradf);
  pair r = inwardnormal(s, cuv, edge);
  if (!inward) r = -r;
  pair d = (g.y, -g.x)
      + crop3_settings.regularizer * abs(gradf) * max(abs(J[0]), abs(J[1])) * r;
  return d == (0,0) ? r : d;
}

// A path across the interior, from an exit crossing back to an entry crossing.
private cropedge cutedge(patch s, pair[] cuv, cropvertex v1, int edge1,
                         cropvertex v2, int edge2, triple grad(triple)) {
  pair d1 = cutdirection(s, cuv, v1, edge1, true, grad(v1.position));
  pair d2 = cutdirection(s, cuv, v2, edge2, false, grad(v2.position));
  return lift(s, v1, v2,
              regular(s, v1.uv) {d1} .. {d2} regular(s, v2.uv));
}

// One Bezier triangle or patch whose side normals match those of the patch
// being cropped.
private patch onepatch(patch s, cropedge[] e) {
  int m = e.length;
  assert(m == 3 || m == 4);
  path3 external = e[0].g;
  for (int i = 1; i < m; ++i) external = external & e[i].g;
  external = external & cycle;

  patch toreturn;
  if (m == 3) {
    toreturn = trianglewithnormals(external, normalat(s, e[0], 0.5),
                                   normalat(s, e[1], 0.5),
                                   normalat(s, e[2], 0.5));
  } else {
    // patchwithnormals samples sides at i/4, 1 + i/4, 3 - i/4 and 4 - i/4, so
    // the last two run backwards.
    triple[] side(int i, bool forward) {
      return sequence(new triple(int q) {
          real t = (q + 1) / 4;
          return normalat(s, e[i], forward ? t : 1 - t);
        }, 3);
    }
    toreturn = patchwithnormals(external, u0normals=side(3, false),
                                u1normals=side(1, true),
                                v0normals=side(0, true),
                                v1normals=side(2, false));
  }
  if (s.colors.length != 0)
    toreturn.colors = sequence(new pen(int i) {
        return colorat(s, e[i].uv0);
      }, m);
  if (s.planar) toreturn.planar = true;
  return toreturn;
}

/***********************************************/
/************ THE CENTRE OF A FAN **************/
/***********************************************/
// A fan decomposes its region only if every spoke stays inside it and no two
// spokes meet away from the centre. The average of the corners need not manage
// either, so the centre has to be searched for; see tests/crop/crop3.asy. The
// region's domain preimage is an ordinary cyclic path, so the search below is
// just `inside` and `intersections`.

// Project d into the cone of nonnegative combinations of a and b, keeping a
// little of each so it cannot run along either: the two-dimensional counterpart
// of smoothcontour3's projecttospan. A reflex corner constrains nothing.
private pair projecttocone(pair d, pair a, pair b, real mincoeff = 0.05) {
  a = unit(a);
  b = unit(b);
  real det = a.x*b.y - a.y*b.x;
  if (det <= 0) return d;
  d = unit(d);
  real alpha = (d.x*b.y - d.y*b.x) / det;
  real beta = (a.x*d.y - a.y*d.x) / det;
  return max(alpha, mincoeff)*a + max(beta, mincoeff)*b;
}

// The spoke of a fan, in the domain, from the centre c to corner u, bent into
// the wedge between the two boundary edges at that corner so as to arrive from
// inside the region. It makes no attempt to steer around the far side of the
// region; the search below rejects such a spoke instead.
private path spokepath(path region, pair c, int u) {
  pair w = point(region, u);
  pair into = projecttocone(c - w,
                            dir(region, u, sign=1),     // on along the boundary
                            -dir(region, u, sign=-1));  // back along it
  return c {w - c} .. {-into} w;
}

// How many corners a fan centred at c fails to reach cleanly: its spoke hits
// the boundary on the way, or crosses a neighbouring spoke, folding the
// triangle between them over. A centre outside reaches none, so this is zero
// iff the fan is valid.
private int unseen(path region, pair c) {
  int n = length(region);
  if (!inside(region, c)) return n;
  path[] spoke = sequence(new path(int u) { return spokepath(region, c, u); },
                          n);
  bool[] fails = array(n, false);
  for (int u = 0; u < n; ++u) {
    for (real[] x : intersections(spoke[u], region))
      if (x[0] < 1 - 1e-4) { fails[u] = true; break; }
    // Spokes always meet at the centre; meeting anywhere else is a fold, and is
    // invisible to the test above, since neither spoke need leave the region.
    // Only neighbours are compared: reaching a spoke further round means
    // crossing a neighbouring spoke or the boundary on the way.
    int w = (u+1) % n;
    for (real[] x : intersections(spoke[u], spoke[w]))
      if (x[0] > 1e-4 && x[1] > 1e-4) {
        fails[u] = true;
        fails[w] = true;
        break;
      }
  }
  int toreturn = 0;
  for (bool b : fails) if (b) ++toreturn;
  return toreturn;
}

// How far a centre stands off the boundary: one that hugs it makes slivers even
// when the fan is valid, so this is maximized.
private real clearance(path region, pair c) {
  real toreturn = realMax;
  for (int i = 0; i < length(region); ++i)
    for (int k = 0; k < 8; ++k)
      toreturn = min(toreturn, abs(point(region, i + k/8) - c));
  return toreturn;
}

// Interior points found by following the chord from c to each corner. Sampling
// the whole of each passage, rather than its midpoint, lets one round reach
// deep into a corner, where the kernel of a region bitten into by its own cut
// tends to lie.
private pair[] chordpoints(path region, pair c) {
  static int SAMPLES = 8;
  pair[] toreturn;
  for (int u = 0; u < length(region); ++u) {
    path chord = c--point(region, u);
    real[] t = {0, 1};
    for (real[] x : intersections(chord, region)) t.push(x[0]);
    t = sort(t);
    for (int i = 0; i + 1 < t.length; ++i) {
      if (t[i+1] - t[i] < 1e-3) continue;   // a grazing contact, not a passage
      for (int k = 1; k < SAMPLES; ++k) {
        pair p = point(chord, interp(t[i], t[i+1], k/SAMPLES));
        if (inside(region, p)) toreturn.push(p);
      }
    }
  }
  return toreturn;
}

// The centre to fan from: the average of the corners if every spoke from it
// arrives, else the best chord point, ranked by spokes that fail and then by
// clearance. The search repeats from its own best answer, since one round need
// not land among the centres that work.
private pair fancentre(path region, pair average) {
  static int ROUNDS = 8;
  pair best = average;
  int bestunseen = unseen(region, average);
  real bestclearance = clearance(region, average);

  for (int round = 0; round < ROUNDS && bestunseen > 0; ++round) {
    pair start = best;
    for (pair c : chordpoints(region, start)) {
      int miss = unseen(region, c);
      real room = clearance(region, c);
      if (miss != bestunseen ? miss < bestunseen : room > bestclearance) {
        best = c;
        bestunseen = miss;
        bestclearance = room;
      }
    }
    if (best == start) break;   // the round found nothing better
  }

  if (bestunseen > 0)
    write('Warning: crop3 cannot place the centre of a fan so that its spokes '
          + 'reach every corner of the region without crossing; the patches '
          + 'covering it will overlap.');
  return best;
}

/***********************************************/
/*********** FILLING WITH A FAN ****************/
/***********************************************/

// Fill a region of any number of sides with a fan of Bezier triangles meeting
// at one interior point. Each spoke is shared as an identical curve by the
// triangles either side of it, and each side of the region becomes a whole side
// of one triangle: so the fan neither cracks internally nor subdivides a
// boundary segment, which would leave a T-junction against the neighbour.
private patch[] fanpatches(patch s, cropvertex[] v, cropedge[] e) {
  int m = e.length;

  // The region in the regularized domain; its nodes are the corners.
  path region = e[0].domain;
  for (int i = 1; i < m; ++i) region = region & e[i].domain;
  region = region & cycle;
  pair average = (0, 0);
  for (int i = 0; i < m; ++i) average += point(region, i);
  average /= m;

  pair x = fancentre(region, average);
  cropvertex centre = cropvertex(s, native(s, x));

  cropedge[] spoke = sequence(new cropedge(int u) {
      return lift(s, centre, v[u], spokepath(region, x, u));
    }, m);

  patch[] toreturn;
  for (int u = 0; u < m; ++u) {
    int w = (u+1) % m;
    toreturn.push(onepatch(s, new cropedge[]
                           {e[u], reverse(spoke[w]), spoke[u]}));
  }
  return toreturn;
}

private patch[] fillregion(patch s, cropvertex[] v, cropedge[] e) {
  if (e.length <= 4 && !crop3_settings.alwaysfan)
    return new patch[] {onepatch(s, e)};
  return fanpatches(s, v, e);
}

/***********************************************/
/**************** CROPPING *********************/
/***********************************************/

// Approximate the part of a Bezier patch or triangle on which f >= 0 by a
// collection of Bezier patches; grad defaults to a numerical approximation.
//
// The zero set is sought only along the patch boundary, and only where that
// changes sign between corners: with every corner on the same side, the patch
// is kept or discarded whole. To resolve a zero set that enters and leaves
// through one edge, or cuts off a region holding no corner, subdivide first.
patch[] crop(patch s, real f(triple), triple grad(triple) = nGrad(f))
{
  path3 e = s.external();
  pair[] cuv = cornerdomain(s);
  int n = cuv.length;

  real[] fv = sequence(new real(int i) { return f(point(e, i)); }, n);
  bool[] keep = sequence(new bool(int i) { return fv[i] >= 0; }, n);
  int nkeep = 0;
  for (bool b : keep) if (b) ++nkeep;
  if (nkeep == n) return new patch[] {s};
  if (nkeep == 0) return new patch[0];

  // Where each sign-changing segment of the boundary crosses the zero set.
  real[] crossing = new real[n];
  for (int k = 0; k < n; ++k) {
    int k1 = (k+1) % n;
    if (keep[k] != keep[k1])
      crossing[k] = edgecrossing(e, k, fv[k], fv[k1], f, grad);
  }

  patch[] toreturn;
  for (int k = 0; k < n; ++k) {
    // Start a region where the boundary enters {f >= 0}; follow it to the exit.
    if (keep[k] || !keep[(k+1) % n]) continue;
    int j = (k+1) % n;
    while (keep[(j+1) % n]) j = (j+1) % n;
    int d = (j - k + n) % n;  // number of corners of the patch kept here

    // Times, unwrapped so as to increase, at which the region's boundary is
    // broken up: the entry crossing, the kept corners, the exit crossing. A
    // crossing snapped onto a corner duplicates its breakpoint and is dropped.
    real[] times = {crossing[k]};
    for (int i = 1; i <= d; ++i) times.push(k+i);
    times.push(k + d + crossing[j] - j);
    real[] breaks;
    for (real t : times)
      if (breaks.length == 0 || t > breaks[breaks.length-1]) breaks.push(t);
    // Fewer than two segments survive only for a region slivered away by the
    // snapping; there is nothing to draw.
    if (breaks.length < 3) continue;

    cropvertex[] v = sequence(new cropvertex(int i) {
        return cropvertex(uvattime(cuv, breaks[i]), point(e, breaks[i]));
      }, breaks.length);
    // Consecutive breakpoints lie on one segment of the patch boundary, so each
    // surviving piece is the image of a straight segment of the domain.
    cropedge[] region = sequence(new cropedge(int i) {
        return cropedge(subpath(e, breaks[i], breaks[i+1]),
                        regular(s, v[i].uv) -- regular(s, v[i+1].uv),
                        v[i].uv, v[i+1].uv);
      }, breaks.length - 1);
    // Close the region with a cut across the interior, exit back to entry.
    int last = v.length - 1;
    region.push(cutedge(s, cuv, v[last], j, v[0], k, grad));

    toreturn.append(fillregion(s, v, region));
  }
  return toreturn;
}

// The part of a surface on which f >= 0, cropping each patch independently.
surface crop(surface s, real f(triple), triple grad(triple) = nGrad(f))
{
  patch[] toreturn;
  for (patch p : s.s) toreturn.append(crop(p, f, grad));
  return surface(... toreturn);
}

/***********************************************/
/************** COMMON REGIONS *****************/
/***********************************************/

// A scalar field paired with its gradient: the region {f >= 0} to crop to.
struct cropfield {
  real f(triple);
  triple grad(triple);
  void operator init(real f(triple), triple grad(triple) = nGrad(f)) {
    this.f = f;
    this.grad = grad;
  }
}

patch[] crop(patch s, cropfield c) { return crop(s, c.f, c.grad); }
surface crop(surface s, cropfield c) { return crop(s, c.f, c.grad); }

// Crop to the intersection of several regions, by cropping to each in turn.
// Not the same as cropping to the minimum of the fields, and the one that
// works: the minimum is creased wherever two meet, and a single cut across a
// patch cannot follow a crease. One region at a time puts each crease on a
// boundary some patch already has, where the next crop meets it exactly.
patch[] crop(patch s, cropfield[] c) {
  patch[] toreturn = {s};
  for (cropfield region : c) {
    patch[] next;
    for (patch q : toreturn) next.append(crop(q, region));
    toreturn = next;
  }
  return toreturn;
}

surface crop(surface s, cropfield[] c) {
  for (cropfield region : c) s = crop(s, region);
  return s;
}

// The companion of the builtin all.
private bool any(bool[] a) { return !all(!a); }


private bool keepabove(string relation) {
  // Strict and non-strict crop alike, differing only on the zero set itself;
  // only the backslashed TeX names need a raw string.
  static string[] ATLEAST_NAMES = {'>=', '>', '≥', 'geq', 'ge', "\geq", "\ge"};
  static string[] ATMOST_NAMES = {'<=', '<', '≤', 'leq', 'le', "\leq", "\le"};
  if (any(ATLEAST_NAMES == relation)) return true;
  if (any(ATMOST_NAMES == relation)) return false;
  abort('crop3: unrecognized relation "' + relation
        + '"; expected >= or <= (or >, <, ge, geq, le, leq).');
  return true;
}

// The half space on one side of the plane dot(p, normal) == value. The normal
// need not be a unit vector, but the bound is read in its scale, not as a
// distance: scaling the normal scales the bound that names the same plane.
cropfield halfspace(triple normal, string relation, real value) {
  bool above = keepabove(relation);
  triple n = above ? normal : -normal;
  real v = above ? value : -value;
  return cropfield(new real(triple p) { return dot(p, n) - v; },
                   new triple(triple p) { return n; });
}

// The six half spaces whose intersection is the axis-aligned box with opposite
// corners a and b.
cropfield[] boxsides(triple a, triple b) {
  triple lo = minbound(a, b), hi = maxbound(a, b);
  return new cropfield[] {
    halfspace(X, '>=', lo.x), halfspace(X, '<=', hi.x),
    halfspace(Y, '>=', lo.y), halfspace(Y, '<=', hi.y),
    halfspace(Z, '>=', lo.z), halfspace(Z, '<=', hi.z)
  };
}

// The inside -- or, with '>=', the outside -- of a sphere. A difference of
// squares rather than the signed distance: same zero set, but differentiable at
// the centre too.
cropfield ball(triple centre, real radius, string relation='<=') {
  real side = keepabove(relation) ? 1 : -1;   // +1 keeps the outside
  real r2 = radius^2;
  return cropfield(new real(triple p) {
      return side * (abs2(p - centre) - r2);
    }, new triple(triple p) { return (2*side) * (p - centre); });
}

// The inside -- or, with '>=', the outside -- of an axis-aligned ellipsoid.
cropfield ellipsoid(triple centre, triple radii, string relation='<=') {
  assert(radii.x != 0 && radii.y != 0 && radii.z != 0,
         'crop3: an ellipsoid needs three nonzero semi-axes.');
  real side = keepabove(relation) ? 1 : -1;
  triple w = (1/radii.x^2, 1/radii.y^2, 1/radii.z^2);
  return cropfield(new real(triple p) {
      triple d = p - centre;
      return side * (w.x*d.x^2 + w.y*d.y^2 + w.z*d.z^2 - 1);
    }, new triple(triple p) {
      triple d = p - centre;
      return (2*side) * (w.x*d.x, w.y*d.y, w.z*d.z);
    });
}

// Crop straight to one of these regions. A box has no such spelling, since two
// triples would not say whether they meant opposite corners or a centre and
// semi-axes: write crop(s, boxsides(a, b)).
patch[] crop(patch s, triple normal, string relation, real value) {
  return crop(s, halfspace(normal, relation, value));
}
surface crop(surface s, triple normal, string relation, real value) {
  return crop(s, halfspace(normal, relation, value));
}
patch[] crop(patch s, triple centre, real radius, string relation='<=') {
  return crop(s, ball(centre, radius, relation));
}
surface crop(surface s, triple centre, real radius, string relation='<=') {
  return crop(s, ball(centre, radius, relation));
}
patch[] crop(patch s, triple centre, triple radii, string relation='<=') {
  return crop(s, ellipsoid(centre, radii, relation));
}
surface crop(surface s, triple centre, triple radii, string relation='<=') {
  return crop(s, ellipsoid(centre, radii, relation));
}
