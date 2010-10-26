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
  Int numLocals;

#ifdef DEBUG_FRAME
  string name;
#endif

public:
  frame(string name, frame *parent, size_t numFormals)
    : parent(parent), numFormals(numFormals), numLocals(0)
#ifdef DEBUG_FRAME
      , name(name)
#endif
  
  {}

  size_t getNumFormals() {
    return numFormals;
  } 
  Int getNumLocals() {
    return numLocals;
  }

  frame *getParent() {
    return parent;
  }

  // Which variable stores the link to the parent frame.
  Int parentIndex() {
    return 0;
  }

  Int size() {
    return (Int) (1+numFormals+numLocals);
  }

  access *accessFormal(size_t index) {
    assert(index < numFormals);
    return new localAccess((Int) (1 + index), this);
  }

  access *allocLocal() {
    return new localAccess((Int) (1 + numFormals + numLocals++), this);
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

inline void print(ostream& out, frame *f) {
  out << f;
  if (f != 0) {
    out << " -> ";
    print(out, f->getParent());
  }
}

} // namespace trans

#endif
     
