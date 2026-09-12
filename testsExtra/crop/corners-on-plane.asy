// All four corners exactly on the crop plane.  KNOWN FAILURE: leaks 14.03.
//
// Run from this directory: asy -dir ../../base corners-on-plane.asy
//
// The teapot is symmetric about y = 0, and the spout is built from patches
// whose four corners all sit exactly on that plane while the body of the patch
// bulges to y = -14. Cropping to y >= 0 finds no sign change between any pair
// of corners, so the patch is kept whole and half of it ends up on the wrong
// side.
//
// This is the boundary-only limitation documented on crop(): the zero set is
// sought along the patch boundary alone. Subdividing the patch once fixes it
// completely, since the halves do have sign-changing corners. It is recorded
// here because the symmetry plane of a symmetric model is among the most
// natural things to crop by, and it fails silently rather than loudly.

import teapotpatches;

patch[] out = crop(patch(spoutbody), y, ygrad);
assert(out.length == 1);   // kept whole, as the limitation predicts
expect('spout cropped to y >= 0', out, y, ygrad);

write('passed -- update or retire this file');
