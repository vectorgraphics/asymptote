/*****
 * drawpath3.cc
 *
 * Stores a path3 that has been added to a picture.
 *****/

#include "drawpath3.h"

namespace camp {

bool drawPath3::write(prcfile *out)
{
  if(n == 0 || pentype.invisible())
    return true;

  pentype.torgb();
  
  RGBAColour rgba(pentype.red(),pentype.green(),pentype.blue(),
		  pentype.opacity());

  if(straight)
    out->add(new PRCline(out,n,controls,rgba));
  else
    out->add(new PRCBezierCurve(out,3,n,controls,rgba));
  
  return true;
}

} //namespace camp
