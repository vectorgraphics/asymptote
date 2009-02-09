/*****
 * guide.cc
 * Andy Hammerlindl 2005/02/23
 *
 *****/

#include "guide.h"

namespace camp {

void multiguide::flatten(flatguide& g, bool allowsolve)
{
  size_t n=v.size();
  if(n > 0) {
    for(size_t i=0; i+1 < n; ++i) {
      v[i]->flatten(g,allowsolve);
      if(!allowsolve && v[i]->cyclic()) {
        g.precyclic(true);
        g.resolvecycle();
      }
    }
    v[n-1]->flatten(g,allowsolve);
  }
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
