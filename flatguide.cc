/*****
 * flatguide.cc
 * Andy Hammerlindl 2005/02/23
 *
 * The data structure that builds up a knotlist.  This is done by calling in
 * order the methods to set knots, specifiers, and tensions.
 * Used by the guide solving routines.
 *****/

#include "flatguide.h"

namespace camp {

void flatguide::addPre(path& p, Int j)
{
  setSpec(new controlSpec(p.precontrol(j),p.straight(j-1)),IN);
}
void flatguide::addPoint(path& p, Int j)
{
  add(p.point(j));
}
void flatguide::addPost(path& p, Int j)
{
  setSpec(new controlSpec(p.postcontrol(j),p.straight(j)),OUT);
}

void flatguide::uncheckedAdd(path p)
{
  Int n=p.length();
  if (n>=0)
    addPoint(p,0);
  for (Int i=1; i<=n; ++i) {
    addPost(p,i-1);
    addPre(p,i);
    addPoint(p,i);
  }
}

spec flatguide::open;

}
