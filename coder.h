/*****
 * coder.h
 * Andy Hammerlindl 2004/11/06
 *
 * Handles encoding of syntax into programs.  It's methods are called by
 * abstract syntax objects during translation to construct the virtual machine
 * code.
 *****/

#ifndef CODER_H
#define CODER_H

#include <list>
#include <map>
#include <stack>

#include "errormsg.h"
#include "entry.h"
#include "types.h"
#include "cast.h"
#include "record.h"
#include "frame.h"
#include "inst.h"
#include "import.h"
#include "util.h"

namespace trans {

using std::list;

using sym::symbol;
using types::ty;
using types::function;
using types::record;

using vm::bltin;
using vm::inst;
using vm::item;

enum modifier {
  DEFAULT_STATIC,
  DEFAULT_DYNAMIC,
  EXPLICIT_STATIC,
  EXPLICIT_DYNAMIC
};

class coder {
  // The frame of the function we are currently encoding.  This keeps
  // track of local variables, and parameters with respect to the stack.
  frame *level;

  // The frame of the enclosing record that the "this" expression yields.  ie.
  // the highest frame that is a record, not a function. 
  frame *recordLevel;

  // The type of the enclosing record.  Also needed for the "this" expression.
  record *recordType;
  
  // The lambda being constructed. In some cases, this lambda is needed
  // before full translation of the function, so it is stored,
  // incomplete, here.
  vm::lambda *l;

  // The type of the function being translated.
  const function *funtype;

  // The enclosing environment.  Null if this is a file-level module.
  coder *parent;

  // The mode of encoding, either static or dynamic. sord is used as an
  // acronym for Static OR Dynamic.
  // Once something is static, no amount of dynamic modifiers can change
  // that, so once a stack is EXPLICIT_STATIC, additional modifiers will
  // be pushed on as EXPLICIT_STATIC.
  modifier sord;
  std::stack<modifier> sord_stack;

  // What permissions will be given to a new access.
  // TODO: Ensure private fields don't show up calling lookup for a
  // record.
  permission perm;
  
  // The function code as its being written.  Code points to next place in
  // array to write.
  // NOTE: As the program array is only given a finite size at its
  // allocation, it may overflow for a large program.  Check on fixing
  // this.
  vm::program program;

  // Keeps track of labels and writes in memory addresses as they're defined.
  // This way a label can be used before its address is known.
  std::map<int,vm::program::label> defs;
  std::multimap<int,vm::program::label> uses;
  int numLabels;

  // Loops need to store labels to where break and continue statements
  // should pass control to.  Since loops can be nested, this needs to
  // be stored as a stack.
  std::stack<int> breakLabels;
  std::stack<int> continueLabels;

  // Current File Position
  position curPos;
private:
  // Define a new function coder.
  coder(function *t, coder &parent, modifier sord = DEFAULT_DYNAMIC);

  // Start encoding the body of the record.  The function being encoded
  // is the record's initializer.
  coder(record *t, coder &parent, modifier sord = DEFAULT_DYNAMIC);

public:
  // Start encoding the body of a file-level record.
  coder(modifier sord = DEFAULT_DYNAMIC);
  
  coder(const coder&);
  
  /* Add a static or dynamic modifier. */
  void pushModifier(modifier s)
  {
    /* Default setting should only be used in the constructor. */
    assert(s != DEFAULT_STATIC && s != DEFAULT_DYNAMIC);

    /* Non-default static overrules. */
    if (sord != EXPLICIT_STATIC)
      sord = s;

    sord_stack.push(sord);
  }

  /* Tests if encoding mode is currently static. */
  bool isStatic()
  {
    switch(sord) {
      case DEFAULT_STATIC:
      case EXPLICIT_STATIC:
	return true;
      case DEFAULT_DYNAMIC:
      case EXPLICIT_DYNAMIC:
	return false;
      default:
	assert(False);
	return false;
    }
  }


  /* Remove a modifier. */
  void popModifier()
  {
    assert(!sord_stack.empty());
    sord_stack.pop();

    assert(!sord_stack.empty());
    sord = sord_stack.top();
  }

  /* Set/get/clear permissions. */
  void setPermission(permission p)
  {
    perm = p;
  }
  permission getPermission()
  {
    return perm;
  }
  void clearPermission()
  {
    perm = READONLY;
  }
    

  // Says what the return type of the function is.
  ty *getReturnType() {
    return funtype->result;
  }

  bool isRecord();
  
  // Creates a new coder to handle the translation of a new function.
  coder newFunction(function *t, modifier sord=DEFAULT_DYNAMIC);

  // Creates a new record type.
  record *newRecord(symbol *id);

  // Create a coder for the initializer of the record.
  coder newRecordInit(record *r, modifier sord=DEFAULT_DYNAMIC);


  frame *getFrame()
  {
    if (isStatic()) {
      assert(parent);
      return parent->getFrame();
    }
    else
      return level;
  }

  // Allocates space in the function or record frame for a new local variable.
  access *allocLocal(permission p)
  {
    return getFrame()->allocLocal(p);
  }

  access *allocLocal()
  {
    return allocLocal(perm);
  }

  // Get the access in the frame for a specified formal parameter.
  access *accessFormal(int index)
  {
    return level->accessFormal(index);
  }


  // The encode functions add instructions and operands on to the code array.
  void encode(inst i)
  {
    i.pos = curPos;
    curPos = position::nullPos();
    if (isStatic()) {
      assert(parent);
      parent->encode(i);
    }
    else {
      program.encode(i);
    }
  }

  void encode(inst::opcode op)
  {
    inst i; i.op = op;
    encode(i);
  }

  void encode(inst::opcode op, int val)
  {
    inst i; i.op = op; i.val = val;
    encode(i);
  }
  void encode(inst::opcode op, item it)
  {
    inst i; i.op = op; i.ref = it;
    encode(i);
  }
  void encode(inst::opcode op, bltin func)
  {
    
    inst i; i.op = op; i.bfunc = func;
    encode(i);
  }
  void encode(inst::opcode op, vm::lambda *l)
  {
    inst i; i.op = op; i.lfunc = l;
    encode(i);
  }

  // Puts tphe requested frame on the stack.  If the frame is not that of
  // this coder or its ancestors, false is returned.
  bool encode(frame *f);

  // Puts the frame corresponding to the expression "this" on the stack.
  bool encodeThis()
  {
    assert(recordLevel);
    return encode(recordLevel);
  }

  // An access that encodes the frame corresponding to "this".
  access *thisLocation()
  {
    assert(recordLevel);
    return new frameAccess(recordLevel);
  }

  // Returns the type of the enclosing record.
  record *thisType()
  {
    return recordType;
  }

  // Puts the 'dest' frame on the stack, assuming the frame 'top' is on
  // top of the stack.  If 'dest' is not a ancestor frame of 'top',
  // false is returned.
  bool encode(frame *dest, frame *top);


  // Assigns a handle to the current point in the list of stack
  // instructions and returns that handle.
  int defLabel();

  // Sets the handle given by label to the current point in the list of
  // instructions.
  int defLabel(int label);

  // Encodes the address pointed to by the handle label into the
  // sequence of instructions.  This is useful for jump instruction to
  // jump to where a label was defined.
  void useLabel(inst::opcode op, int label);

  // If an address has to be used for a jump instruction before it is
  // actually encoded, a handle can be given to it by this function.
  // When that handle's label is later defined, the proper address will
  // be inserted into the code where the handle was used. 
  int fwdLabel();

  void pushBreak(int label) {
    breakLabels.push(label);
  }
  void pushContinue(int label) {
    continueLabels.push(label);
  }
  void popBreak() {
    breakLabels.pop();
  }
  void popContinue() {
    continueLabels.pop();
  }
  bool encodeBreak() {
    if (breakLabels.empty())
      return false;
    else {
      useLabel(inst::jmp,breakLabels.top());
      return true;
    }
  }
  bool encodeContinue() {
    if (continueLabels.empty())
      return false;
    else {
      useLabel(inst::jmp,continueLabels.top());
      return true;
    }
  }
  
  // Adds an entry into the position list, linking the given point in the
  // source code to the current position in the virtual machine code.  This is
  // used to print positions at runtime.
  void markPos(position pos);

  // When translating the function is finished, this ties up loose ends
  // and returns the lambda.
  vm::lambda *close();

private: // Non-copyable
  void operator=(const coder&);
};

} // namespace trans

#endif
