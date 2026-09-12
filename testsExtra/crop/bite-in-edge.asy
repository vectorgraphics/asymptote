// The zero set enters and leaves through a single edge.  KNOWN FAILURE:
// leaks 0.46.
//
// Run from this directory: asy -dir ../../base bite-in-edge.asy
//
// The oblique plane passes just inside one edge of this lid patch: along that
// edge f runs 0.91, -0.12, -0.46, -0.12, 0.91, so both endpoints are kept
// while a bite out of the middle should not be. Both corners test positive, no
// crossing is found on that edge, and the region swallows the bite.
//
// The same documented boundary-only limitation as corners-on-plane.asy, and
// the more insidious half of it: there the patch is passed through untouched,
// which is at least easy to spot, whereas here the patch really is cropped and
// the output looks plausible.

import teapotpatches;

patch p = patch(lidquarter);
path3 e = p.external();

// Show the sign change that the corner test cannot see.
for (int k = 0; k < 4; ++k) {
  real a = oblique(point(e, k)), b = oblique(point(e, k+1));
  real lo = min(a, b);
  for (int i = 1; i < 40; ++i) lo = min(lo, oblique(point(e, k + i/40)));
  if (a >= 0 && b >= 0 && lo < 0)
    write('edge ' + string(k) + ': ends ' + format('%.2f', a) + ', '
          + format('%.2f', b) + ' but dips to ' + format('%.2f', lo));
}

expect('lid quarter cropped by oblique plane',
       crop(p, oblique, obliquegrad), oblique, obliquegrad);

write('passed -- update or retire this file');
