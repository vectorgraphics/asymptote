/*****
 * pathlist.h
 * Andy Hammerlindl 2002/08/23
 *
 * As segments are solved in a guide, they are put into this structure.
 * Once the entire guide is solved, then the segments of path are
 * intelligently concatenated into a single path.  One important thing
 * to address is that a cyclic path has its indices in the proper order.
 *****/

#include "pathlist.h"

namespace camp {

pathlist::~pathlist()
{
}
 
void pathlist::add(path p)
{
  paths.push_back(p);
}

void pathlist::addToFront(path p)
{
  paths.push_front(p);
}

void pathlist::add(path p, int offset)
{
  paths.push_back(p.subpath(0, offset));
  paths.push_front(p.subpath(offset, p.length()));
} 

path pathlist::solve()
{
  // The individual paths have to be joined.  Hopefully, the last
  // knot of one path will be the first knot of the next.
  if (paths.size() == 3
      && paths.front().length() == 0
      && paths.back().length() == 0)
    return *++paths.begin();
 
  // First count the number of knots. This accounts for shared knots.
  int n = 0;
  for (list<path>::iterator p = paths.begin(); p != paths.end(); ++p) {
    n += p->length();
  }
  // Non-cyclic paths need a point for the last position.
  if (!cyclic) n++;

  solvedKnot *nodes = new solvedKnot[n];

  int i = 0;
  for (list<path>::iterator p = paths.begin(); p != paths.end(); ++p) {
    int length = p->length();
    
    for (int j = 0; j < length; j++) {
      nodes[i].point = p->point(j);
      nodes[i].post = p->postcontrol(j);
      nodes[i].straight=p->straight(j);
      if (i+1 < n)
        nodes[i+1].pre = p->precontrol(j+1);
      else
        nodes[0].pre = p->precontrol(j+1);
      i++;
    }
  }

  if (!cyclic) {
    nodes[0].pre = nodes[0].point;

    path p = paths.back();
    nodes[n-1].point = nodes[n-1].post = p.point(p.length());
  }

  return path(nodes, n, cyclic);
}

} // namespace camp
