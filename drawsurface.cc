/*****
 * drawsurface.cc
 *
 * Stores a surface that has been added to a picture.
 *****/

#include "drawsurface.h"

namespace camp {

bool drawSurface::write(prcfile *out)
{
  if(pentype.invisible() || pentype.width() == 0.0)
    return true;

  pentype.torgb();
  
  RGBAColour rgba(pentype.red(),pentype.green(),pentype.blue(),
		  pentype.opacity());

  out->add(new PRCBezierSurface(out,3,3,4,4,controls,rgba));
  
  return true;
}

} //namespace camp
