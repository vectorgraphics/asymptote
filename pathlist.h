/*****
 * pathlist.h
 * Andy Hammerlindl 2002/08/23
 *
 * As segments are solved in a guide, they are put into this structure.
 * Once the entire guide is solved, then the segments of path are
 * intelligently concatenated into a single path.  One important thing
 * to address is that a cyclic path has its indices in the proper order.
 *****/

#ifndef PATHLIST_H
#define PATHLIST_H

#include <list>
#include "path.h"

namespace camp {

using std::list;
 
class pathlist {
  list<path> paths;
  bool cyclic;

public:
  pathlist()
    : cyclic(false) {}

  virtual ~pathlist();
  
  // Adds a path to the end
  void add(path p);

  // Adds a path to the front.
  void addToFront(path p);

  // Adds a path, the first offset sections to the end and the rest to
  // the front.
  void add(path p, int offset);

  // Sets whether or not the path is cyclic.
  void setCyclic(bool state) {
    cyclic = state;
  }
  
  // Returns the formed path.
  path solve();

};

} //namespace camp

#endif
