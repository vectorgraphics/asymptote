/*****
 * access.h
 * Andy Hammerlindl 2003/12/03
 *
 * Describes an "access," a representation of where a variable will be
 * stored at runtime, so that read, write, and call instructions can be
 * made.
 *****/

#ifndef ACCESS_H
#define ACCESS_H

#include <cassert>

#include "errormsg.h"
#include "pool.h"
#include "inst.h"

using vm::inst;
using vm::bltin;

namespace trans {
  
class frame;
class coder;

// PUBLIC, PRIVATE, or READONLY - the permission tokens defined in
// camp.y for accessing a variable outside of its lexically enclosing
// record.
enum permission {
  READONLY,
  PUBLIC,
  PRIVATE
};


// These serves as the base class for the accesses.
class access : public mempool::pooled<access> {
protected:
  // Generic compiler access error - if the compiler functions properly,
  // none of these should be reachable by the user.
  void error(position pos)
  {
    em->compiler(pos);
    *em << "invalid use of access";
  }

public:
  virtual ~access() = 0;
  
  // Encode a read of the access when nothing is on the stack.
  virtual void encodeRead(position pos, coder &)
  {
    error(pos);
  }
  // Encode a read of the access when the frame "top" is on top
  // of the stack.
  virtual void encodeRead(position pos, coder &, frame *)
  {
    error(pos);
  }

  virtual void encodeWrite(position pos, coder &)
  {
    error(pos);
  }
  virtual void encodeWrite(position pos, coder &, frame *)
  {
    error(pos);
  }
  virtual void encodeCall(position pos, coder &)
  {
    error(pos);
  }
  virtual void encodeCall(position pos, coder &, frame *)
  {
    error(pos);
  }
};

// This class represents identity conversions in casting.
class identAccess : public access 
{
  virtual void encodeCall(position, coder&);
};

// This access represents functions that are implemented by instructions
// in the virtual machine.
class instAccess : public access {
  inst i;

public:
  instAccess(inst i)
    : i(i) {}

  instAccess(inst::opcode o)
    { i.op = o; }

  void encodeCall(position pos, coder &e);
};

// Represents a function that is implemented by a built-in C++ function.
class bltinAccess : public access {
  bltin f;

public:
  bltinAccess(bltin f)
    : f(f) {}

  void encodeRead(position pos, coder &e);
  void encodeRead(position pos, coder &e, frame *top);
  void encodeWrite(position pos, coder &e);
  void encodeWrite(position pos, coder &e, frame *top);
  void encodeCall(position pos, coder &e);
};

// Represents the access of a global variable.
// Not used, as global variables are now represented as a record.
class globalAccess : public access {
  int offset;

public:
  globalAccess(int offset)
    : offset(offset) {}

  void encodeRead(position pos, coder &e);
  //void encodeRead(position pos, coder &e, frame *top);
  void encodeWrite(position pos, coder &e);
  //void encodeWrite(position pos, coder &e, frame *top);
  void encodeCall(position pos, coder &e);
  //void encodeCall(position pos, coder &e, frame *top);
};

// An access that puts a frame on the top of the stack.
class frameAccess : public access {
  frame *f;

public:
  frameAccess(frame *f)
    : f(f) {}
  
  void encodeRead(position pos, coder &e);
  void encodeRead(position pos, coder &e, frame *top);
};

// Represents the access of a local variable.
class localAccess : public access {
  int offset;
  frame *level;

  permission perm;

  /* In the case where we are not in the access's local scope, this
   * checks if permissions are valid for a read/write/call.  Reports an
   * error if such a thing is not allowed.
   */
  void permitRead(position pos);
  void permitWrite(position pos);

public:
  localAccess(permission perm, int offset, frame *level)
    : offset(offset), level(level), perm(perm) {}

  void encodeRead(position pos, coder &e);
  void encodeRead(position pos, coder &e, frame *top);
  void encodeWrite(position pos, coder &e);
  void encodeWrite(position pos, coder &e, frame *top);
  void encodeCall(position pos, coder &e);
  void encodeCall(position pos, coder &e, frame *top);
};

} // namespace trans

#endif // ACCESS_H

