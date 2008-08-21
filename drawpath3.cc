/*****
 * drawpath3.cc
 *
 * Stores a path3 that has been added to a picture.
 *****/

#include "drawpath3.h"

namespace camp {

bool drawPath3::write(prcfile *out)
{
  if(n == 0)
    return true;

  if(straight)
    out->add(new PRCline(out,n,controls,color));
  else
    out->add(new PRCBezierCurve(out,3,n,controls,color));
  
  return true;
}

} //namespace camp
