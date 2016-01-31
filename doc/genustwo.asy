size(10cm,0);
import smoothcontour3;
currentprojection=perspective((18,20,10));
if(settings.render < 0) settings.render=8;

real tuberadius = 0.69;

// Convert to cylindrical coordinates to draw
// a circle revolved about the z axis.
real toruscontour(real x, real y, real z) {
  real r = sqrt(x^2 + y^2);
  return (r-2)^2 + z^2 - tuberadius^2;
}

// Take the union of the two tangent tori (by taking 
// the product of the functions defining them). Then
// add (or subtract) a bit of noise to smooth things 
// out.
real f(real x, real y, real z) {
  real f1 = toruscontour(x - 2 - tuberadius, y, z);
  real f2 = toruscontour(x + 2 + tuberadius, y, z);
  return f1 * f2 - 0.1;
}

// The noisy function extends a bit farther than the union of 
// the two tori, so include a bit of extra space in the box.
triple max = (2*(2+tuberadius), 2+tuberadius, tuberadius)
            + (0.1, 0.1, 0.1);
triple min = -max;

// Draw the implicit surface.
draw(implicitsurface(f, min, max, overlapedges=true, 
                     nx=20, nz=5),
     surfacepen=material(diffusepen=gray(0.6),
			 emissivepen=gray(0.3),
			 specularpen=gray(0.1)));
