/*****
 * guide.cc
 * Andy Hammerlindl 2005/02/23
 *
 *****/

#include "guide.h"

namespace camp {

void multiguide::flatten(flatguide& g)
{
  for (size_t i=0; i<v.size(); ++i)
    v[i]->flatten(g);
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
