/*****
 * access.cc
 * Andy Hammerlindl 2003/12/03
 * Describes an "access," a representation of where a variable will be
 * stored at runtime, so that read, write, and call instructions can be
 * made.
 *****/

#include "access.h"
#include "frame.h"
#include "coder.h"
#include "callable.h"

namespace trans {

/* access */
access::~access()
{}

/* identAccess */
void identAccess::encodeCall(position, coder&)
{}

/* bltinAccess */
static void bltinError(position pos)
{
  em->error(pos);
  *em << "built-in functions cannot be modified";
}

void bltinAccess::encodeRead(position, coder &e)
{
  e.encode(inst::constpush,(item)(vm::callable*)new vm::bfunc(f));
}

void bltinAccess::encodeRead(position, coder &e, frame *)
{
  e.encode(inst::constpush,(item)(vm::callable*)new vm::bfunc(f));
}

void bltinAccess::encodeWrite(position pos, coder &)
{
  bltinError(pos);
}

void bltinAccess::encodeWrite(position pos, coder &, frame *)
{
  bltinError(pos);
}

void bltinAccess::encodeCall(position, coder &e)
{
  e.encode(inst::builtin, f);
}

/* frameAccess */
void frameAccess::encodeRead(position pos, coder &e)
{
  if (!e.encode(f)) {
    em->compiler(pos);
    *em << "encoding frame out of context";
  }
}

void frameAccess::encodeRead(position pos, coder &e, frame *top)
{
  if (!e.encode(f,top)) {
    em->compiler(pos);
    *em << "encoding frame out of context";
  }
}

/* localAccess */
void localAccess::permitRead(position pos)
{
  if (perm == PRIVATE) {
    em->error(pos);
    *em << "accessing private field outside of structure";
  }
}

void localAccess::permitWrite(position pos)
{
  switch (perm) {
    case PRIVATE:
      em->error(pos);
      *em << "modifying private field outside of structure";
      break;
    case READONLY:
      em->error(pos);
      *em << "modifying non-public field outside of structure";
      break;
    case PUBLIC:
      break;
  }
}

void localAccess::encodeRead(position pos, coder &e)
{
  // Get the active frame of the virtual machine.
  frame *active = e.getFrame();
  if (level == active) {
    e.encode(inst::varpush,offset);
  }
  else {
    // Put the parent frame (in local variable 0) on the stack.
    e.encode(inst::varpush,0);

    // Encode the access from that frame.
    this->encodeRead(pos, e, active->getParent());
  }
}

void localAccess::encodeRead(position pos, coder &e, frame *top)
{
  if (e.encode(level,top)) {
    // Test permissions.
    if (!top->isDescendant(e.getFrame()))
      permitRead(pos);
  
    e.encode(inst::fieldpush,offset);
  }
  else {
    // The local variable is being used when its frame is not active.
    em->error(pos);
    *em << "static use of dynamic variable";
  }
}

void localAccess::encodeWrite(position pos, coder &e)
{
  // Get the active frame of the virtual machine.
  frame *active = e.getFrame();
  if (level == active) {
    e.encode(inst::varsave,offset);
  }
  else {
    // Put the parent frame (in local variable 0) on the stack.
    e.encode(inst::varpush,0);

    // Encode the access from that frame.
    this->encodeWrite(pos, e, active->getParent());
  }
}

void localAccess::encodeWrite(position pos, coder &e, frame *top)
{
  if (e.encode(level,top)) {
    // Test permissions.
    if (!top->isDescendant(e.getFrame()))
      permitWrite(pos);
  
    e.encode(inst::fieldsave,offset);
  }
  else {
    // The local variable is being used when its frame is not active.
    em->error(pos);
    *em << "static use of dynamic variable";
  }
}

void localAccess::encodeCall(position pos, coder &e)
{
  encodeRead(pos, e);
  e.encode(inst::popcall);
}

void localAccess::encodeCall(position pos, coder &e, frame *top)
{
  encodeRead(pos, e, top);
  e.encode(inst::popcall);
}

} // namespace trans
