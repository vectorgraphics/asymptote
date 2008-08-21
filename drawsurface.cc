/*****
 * drawsurface.cc
 *
 * Stores a surface that has been added to a picture.
 *****/

#include "drawsurface.h"

namespace camp {

bool drawSurface::write(prcfile *out)
{
  if(invisible)
    return true;

  PRCMaterial m(ambient,diffuse,emissive,specular,opacity,shininess);
  out->add(new PRCBezierSurface(out,3,3,4,4,controls,m,granularity));
  
  return true;
}

} //namespace camp
