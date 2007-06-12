/*****
 * guide.cc
 * Andy Hammerlindl 2005/02/23
 *
 *****/

#include "guide.h"

namespace camp {

bool multiguide::flatten(flatguide& g, bool allowsolve)
{
  size_t n=v.size();
  if(n == 0) return false;
  bool precycle=false;
  for(size_t i=0; i+1 < n; ++i)
    if(v[i]->flatten(g)) precycle=true;
  v[n-1]->flatten(g,allowsolve);
  return precycle;
}

void multiguide::print(ostream& out) const
{
  side lastLoc=JOIN;
  for(size_t i=0; i < v.size(); ++i) {
    side loc=v[i]->printLocation();
    adjustLocation(out,lastLoc,loc);
    v[i]->print(out);
    lastLoc=loc;
  }
}

} // namespace camp
