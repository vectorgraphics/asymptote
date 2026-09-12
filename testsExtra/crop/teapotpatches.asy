// Patches lifted verbatim from examples/teapot.asy, with the measurements the
// crop cases in this directory share. The index in each name is the patch's
// position in the surface that example builds.
//
// Leak is measured as a distance, -f/|grad f|, so that a plane's signed
// distance and a ball's difference of squares are on the same scale. The
// teapot is about 185 units across.

import crop3;

// Patch 16: the underside of the spout.
triple[][] spoutbody = {
  {(48.18897,0,40.3937),(73.70078,0,40.3937),(65.19685,0,59.52755),(76.53543,0,68.0315)},
  {(48.18897,-18.70866,40.3937),(73.70078,-18.70866,40.3937),(65.19685,-7.086619,59.52755),(76.53543,-7.086619,68.0315)},
  {(48.18897,-18.70866,17.00787),(87.87401,-18.70866,23.38582),(68.0315,-7.086619,57.40157),(93.5433,-7.086619,68.0315)},
  {(48.18897,0,17.00787),(87.87401,0,23.38582),(68.0315,0,57.40157),(93.5433,0,68.0315)}
};

// Patch 21: the lid, just off the axis.
triple[][] lidquarter = {
  {(0,-5.669294,76.53543),(0,-11.33858,72.28346),(0,-36.85039,72.28346),(0,-36.85039,68.0315)},
  {(-3.174809,-5.669294,76.53543),(-6.349609,-11.33858,72.28346),(-20.63622,-36.85039,72.28346),(-20.63622,-36.85039,68.0315)},
  {(-5.669294,-3.174809,76.53543),(-11.33858,-6.349609,72.28346),(-36.85039,-20.63622,72.28346),(-36.85039,-20.63622,68.0315)},
  {(-5.669294,0,76.53543),(-11.33858,0,72.28346),(-36.85039,0,72.28346),(-36.85039,0,68.0315)}
};

real leak(patch[] c, real f(triple), triple grad(triple)) {
  real worst = 0;
  for (patch q : c)
    for (int i = 0; i <= 8; ++i)
      for (int j = 0; j <= 8; ++j) {
        real u = i/8, v = j/8;
        if (q.triangular && u + v > 1) continue;
        triple x = q.point(u, v);
        real g = abs(grad(x));
        if (g > 0) worst = max(worst, -f(x) / g);
      }
  return worst;
}

// Report a case, then assert what crop3 ought to manage.
void expect(string name, patch[] out, real f(triple), triple grad(triple),
            real tolerance = 0.05) {
  real lk = leak(out, f, grad);
  write(name + ': ' + string(out.length) + ' patches, leak '
        + format('%.4f', lk));
  assert(lk < tolerance);
}

// The symmetry plane of the teapot.
real y(triple p) { return p.y; }
triple ygrad(triple p) { return Y; }

// A plane at an angle to everything, grazing the lid.
real oblique(triple p) { return (p.x + p.y + p.z)/sqrt(3) - 40; }
triple obliquegrad(triple p) { return unit((1,1,1)); }
