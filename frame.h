/*****
 * frame.h
 * Andy Hammerlindl 2002/07/22
 *
 * Describes the frame levels for the functions of the language.
 * Also creates accesses for the variable for automated loading
 * and saving of variables.
 *****/

#ifndef FRAME_H
#define FRAME_H

#include <cassert>

#include "pool.h"
#include "access.h"

namespace trans {

class frame : public memory::managed<frame> {
  frame *parent;
 
  int numFormals;
  int numLocals;

public:
  frame(frame *parent, int numFormals)
    : parent(parent), numFormals(numFormals), numLocals(0)
  {
    assert(numFormals >= 0);
  }

  int getNumFormals() {
    return numFormals;
  } 
  int getNumLocals() {
    return numLocals;
  }

  frame *getParent() {
    return parent;
  }

  int size() {
    return 1 + numFormals + numLocals;
  }

  access *accessFormal(int index) {
    assert(index >= 0 && index < numFormals);
    return new localAccess(PRIVATE, 1 + index, this);
  }

  access *allocLocal(permission perm) {
    return new localAccess(perm, 1 + numFormals + numLocals++, this);
  }

  // Checks if the frame f is a descendent of this frame.
  // For our purposes, a frame is its own descendant.
  bool isDescendant(frame *f)
  {
    while (f != 0) {
      if (f == this)
	return true;
      f = f->parent;
    }
    return false;
  }
};

} // namespace trans

#endif
     
