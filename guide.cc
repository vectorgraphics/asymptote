/*****
 * guide.cc
 * Andy Hammerlindl 2005/02/23
 *
 *****/

#include "guide.h"

namespace camp {

bool multiguide::flatten(flatguide& g, bool allowsolve)
{
  bool cyclic;
  for (size_t i=0; i<v.size(); ++i)
    cyclic=v[i]->flatten(g,allowsolve);
  return cyclic;
}

void multiguide::print(ostream& out) const
{
  side lastLoc=JOIN;
  for (size_t i=0; i<v.size(); ++i) {
    side loc=v[i]->printLocation();
    adjustLocation(out,lastLoc,loc);
    v[i]->print(out);
    lastLoc=loc;
  }
}

} // namespace camp
