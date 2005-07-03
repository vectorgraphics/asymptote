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

#include "access.h"

namespace trans {

class frame : public gc {
  frame *parent;
 
  size_t numFormals;
  int numLocals;

public:
  frame(frame *parent, size_t numFormals)
    : parent(parent), numFormals(numFormals), numLocals(0) {}

  size_t getNumFormals() {
    return numFormals;
  } 
  int getNumLocals() {
    return numLocals;
  }

  frame *getParent() {
    return parent;
  }

  int size() {
    return numLocals;
  }

  access *accessFormal(size_t index) {
    assert(index < numFormals);
    return new localAccess(PRIVATE, (int) (1 + index), this);
  }

  access *allocLocal(permission perm) {
    return new localAccess(perm, (int) (1 + numFormals + numLocals++), this);
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
     
