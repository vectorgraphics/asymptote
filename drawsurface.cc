/*****
 * drawsurface.cc
 *
 * Stores a surface that has been added to a picture.
 *****/

#include "drawsurface.h"

namespace camp {

bool drawSurface::write(prcfile *out)
{
  if(diffusepen.invisible())
    return true;

  RGBAColour diffuse(diffusepen.red(),diffusepen.green(),diffusepen.blue(),
		     diffusepen.opacity());
  RGBAColour ambient(ambientpen.red(),ambientpen.green(),ambientpen.blue(),
		     ambientpen.opacity());
  RGBAColour emissive(emissivepen.red(),emissivepen.green(),emissivepen.blue(),
		      emissivepen.opacity());
  RGBAColour specular(specularpen.red(),specularpen.green(),specularpen.blue(),
		      specularpen.opacity());
  
  PRCMaterial m(ambient,diffuse,emissive,specular,alpha,shininess);
  out->add(new PRCBezierSurface(out,3,3,4,4,controls,m));
  
  return true;
}

} //namespace camp
