/*****
 * access.h
 * Andy Hammerlindl 2003/12/03
 * Describes an "access," a reprsentation of where a variable will be
 * stored at runtime, so that read, write, and call instructions can be
 * made.
 *****/

#include "access.h"
#include "frame.h"
#include "env.h"

namespace trans {

/* access */
access::~access()
{}

/* identAccess */
void identAccess::encodeCall(position, env&)
{}


/* instAccess */
void instAccess::encodeCall(position, env &e)
{
  e.encode(i);
}


/* bltinAccess */
static void bltinError(position pos)
{
  em->error(pos);
  *em << "built-in functions cannot be modified";
}

void bltinAccess::encodeRead(position, env &e)
{
  e.encode(inst::constpush);
  e.encode((item)(vm::callable*)new vm::bfunc(f));
}

void bltinAccess::encodeRead(position, env &e, frame *)
{
  e.encode(inst::constpush);
  e.encode((item)(vm::callable*)new vm::bfunc(f));
}

void bltinAccess::encodeWrite(position pos, env &)
{
  bltinError(pos);
}

void bltinAccess::encodeWrite(position pos, env &, frame *)
{
  bltinError(pos);
}

void bltinAccess::encodeCall(position, env &e)
{
  e.encode(inst::builtin);
  e.encode(f);
}


/* globalAccess */
void globalAccess::encodeRead(position, env &e)
{
  e.encode(inst::globalpush);
  e.encode(offset);
}

void globalAccess::encodeWrite(position, env &e)
{
  e.encode(inst::globalsave);
  e.encode(offset);
}

void globalAccess::encodeCall(position pos, env &e)
{
  encodeRead(pos, e);
  e.encode(inst::popcall);
}


/* localAccess */
void localAccess::permitRead(position pos)
{
  if (perm == PRIVATE) {
    em->error(pos);
    *em << "accessing private field outside of record";
  }
}

void localAccess::permitWrite(position pos)
{
  switch (perm) {
    case PRIVATE:
      em->error(pos);
      *em << "modifying private field outside of record";
      break;
    case READONLY:
      em->error(pos);
      *em << "modifying non-public field outside of record";
      break;
    case PUBLIC:
      break;
  }
}

void localAccess::encodeRead(position pos, env &e)
{
  // Get the active frame of the virtual machine.
  frame *active = e.getFrame();
  if (level == active) {
    e.encode(inst::varpush);
    e.encode(offset);
  }
  else {
    // Put the parent frame (in local variable 0) on the stack.
    e.encode(inst::varpush);
    e.encode(0);

    // Encode the access from that frame.
    this->encodeRead(pos, e, active->getParent());
  }
}

void localAccess::encodeRead(position pos, env &e, frame *top)
{
  // Test permissions.
  if (!top->isDescendant(e.getFrame()))
    permitRead(pos);
  
  if (top == 0) {
    // The local variable is being used when its frame is not active.
    em->compiler(pos);
    *em << "access used out of scope";
  }
  else if (level == top) {
    e.encode(inst::fieldpush);
    e.encode(offset);
  }
  else {
    // Go another level down.
    e.encode(inst::fieldpush);
    e.encode(0);
    encodeRead(pos, e, top->getParent());
  }
}

void localAccess::encodeWrite(position pos, env &e)
{
  // Get the active frame of the virtual machine.
  frame *active = e.getFrame();
  if (level == active) {
    e.encode(inst::varsave);
    e.encode(offset);
  }
  else {
    // Put the parent frame (in local variable 0) on the stack.
    e.encode(inst::varpush);
    e.encode(0);

    // Encode the access from that frame.
    this->encodeWrite(pos, e, active->getParent());
  }
}

void localAccess::encodeWrite(position pos, env &e, frame *top)
{
  // Test permissions.
  if (!top->isDescendant(e.getFrame()))
    permitWrite(pos);

  if (top == 0) {
    // The local variable is being used when its frame is not active.
    em->compiler(pos);
    *em << "access used out of scope";
  }
  else if (level == top) {
    e.encode(inst::fieldsave);
    e.encode(offset);
  }
  else {
    // Go another level down.
    e.encode(inst::fieldpush);
    e.encode(0);
    encodeWrite(pos, e, top->getParent());
  }
}

void localAccess::encodeCall(position pos, env &e)
{
  encodeRead(pos, e);
  e.encode(inst::popcall);
}

void localAccess::encodeCall(position pos, env &e, frame *top)
{
  encodeRead(pos, e, top);
  e.encode(inst::popcall);
}

} // namespace trans
