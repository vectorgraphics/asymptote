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

#include "errormsg.h"
#include "entry.h"
#include "types.h"
#include "record.h"
#include "frame.h"
#include "program.h"
#include "util.h"
#include "modifier.h"
#include "inst.h"

namespace trans {

using sym::symbol;
using types::ty;
using types::function;
using types::record;

using vm::bltin;
using vm::inst;
using vm::item;

#ifdef DEBUG_BLTIN
void assertBltinLookup(inst::opcode op, item it);
#endif

class coder {
  // The frame of the function we are currently encoding.  This keeps
  // track of local variables, and parameters with respect to the stack.
  frame *level;

  // The frame of the enclosing record that the "this" expression yields.  ie.
  // the highest frame that is a record, not a function. 
  frame *recordLevel;

  // The type of the enclosing record.  Also needed for the "this" expression.
  record *recordType;
  
  // Are we translating a codelet?
  bool isCodelet;

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
  vm::program *program;

  // Keeps track of labels and writes in memory addresses as they're defined.
  // This way a label can be used before its address is known.
  std::map<Int,vm::program::label> defs;
  std::multimap<Int,vm::program::label> uses;
  Int numLabels;

  // The loop constructs allocate nested frames, in case variables in an
  // iteration escape in a closure.  This stack keeps track of where the
  // variables are allocated, so the size of the frame can be encoded.  At the
  // start, it just holds the label of the first instruction of the lambda, as
  // this is where space for the variables of the function is allocated.
  std::stack<vm::program::label> allocs;

  // Loops need to store labels to where break and continue statements
  // should pass control to.  Since loops can be nested, this needs to
  // be stored as a stack.
  std::stack<Int> breakLabels;
  std::stack<Int> continueLabels;

  // Current File Position
  position curPos;

public:
  // Define a new function coder.  If reframe is true, this gives the function
  // its own frame, which is the usual (sensible) thing to do.  It is set to
  // false for a line-at-a-time codelet, where variables should be allocated in
  // the lower frame.
  coder(position pos,
        string name, function *t, coder *parent,
        modifier sord = DEFAULT_DYNAMIC,
        bool reframe=true);

  // Start encoding the body of the record.  The function being encoded
  // is the record's initializer.
  coder(position pos,
        record *t, coder *parent, modifier sord = DEFAULT_DYNAMIC);

  coder(position pos,
        string name, modifier sord = DEFAULT_DYNAMIC);
  
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
    perm = DEFAULT_PERM;
  }
    

  // Says what the return type of the function is.
  ty *getReturnType() {
    return funtype->result;
  }

  bool isRecord();
  
  // Creates a new coder to handle the translation of a new function.
  coder newFunction(position pos,
                    string name, function *t, modifier sord=DEFAULT_DYNAMIC);

  // Creates a new record type.
  record *newRecord(symbol id);

  // Create a coder for the initializer of the record.
  coder newRecordInit(position pos, record *r, modifier sord=DEFAULT_DYNAMIC);

  // Create a coder for translating a small piece of code.  Used for
  // line-at-a-time mode.
  coder newCodelet(position pos);

  frame *getFrame()
  {
    if (isStatic() && !isTopLevel()) {
      assert(parent->getFrame());
      return parent->getFrame();
    }
    else
      return level;
  }

  // Tests if the function or record with the given frame is currently under
  // translation (either by this coder or an ancestor).
  bool inTranslation(frame *f) {
    frame *level=this->level;
    while (level) {
      if (f==level)
        return true;
      level=level->getParent();
    }
    return parent && parent->inTranslation(f);
  }

  // Allocates space in the function or record frame for a new local variable.
  access *allocLocal()
  {
    return getFrame()->allocLocal();
  }

  // Get the access in the frame for a specified formal parameter.
  access *accessFormal(Int index)
  {
    // NOTE: This hasn't been extended to handle frames for loops, but is
    // currently only called when starting to translate a function, where there
    // can be no loops.
    return level->accessFormal(index);
  }

  // Checks if we are at the top level, which is true for a file-level module or
  // a codelet.
  bool isTopLevel() {
    return parent==0 || isCodelet;
  }

  // The encode functions add instructions and operands on to the code array.
private:
  void encode(inst i)
  {
    i.pos = curPos;
    // Static code is put into the enclosing coder, unless we are translating a
    // codelet.
    if (isStatic() && !isTopLevel()) {
      assert(parent);
      parent->encode(i);
    }
    else {
      program->encode(i);
    }
  }

public:
  void encode(inst::opcode op)
  {
    inst i; i.op = op; i.pos = nullPos;
    encode(i);
  }
  void encode(inst::opcode op, item it)
  {
#ifdef DEBUG_BLTIN
    assertBltinLookup(op, it);
#endif
    inst i; i.op = op; i.pos = nullPos; i.ref = it;
    encode(i);
  }

  // Encodes a pop instruction, or merges the pop into the previous
  // instruction (ex. varsave+pop becomes varpop).
  void encodePop();

  // Puts the requested frame on the stack.  If the frame is not that of
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
  // top of the stack.  If 'dest' is not an ancestor frame of 'top',
  // false is returned.
  bool encode(frame *dest, frame *top);


  // Assigns a handle to the current point in the list of stack
  // instructions and returns that handle.
  Int defLabel();

  // Sets the handle given by label to the current point in the list of
  // instructions.
  Int defLabel(Int label);

  // Encodes the address pointed to by the handle label into the
  // sequence of instructions.  This is useful for a jump instruction to
  // jump to where a label was defined.
  void useLabel(inst::opcode op, Int label);

  // If an address has to be used for a jump instruction before it is
  // actually encoded, a handle can be given to it by this function.
  // When that handle's label is later defined, the proper address will
  // be inserted into the code where the handle was used. 
  Int fwdLabel();

  void pushBreak(Int label) {
    breakLabels.push(label);
  }
  void pushContinue(Int label) {
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
  
private:
  void encodeAllocInstruction() {
    encode(inst::alloc, 0);
    allocs.push(--program->end());
  }

  void finishAlloc() {
    allocs.top()->ref = level->size();
    allocs.pop();
  }

public:
  void encodePushFrame() {
    encode(inst::pushframe);
    level = new frame("encodePushFrame", level, 0);

    encodeAllocInstruction();
  }

  void encodePopFrame() {
    finishAlloc();

    encode(inst::popframe);
    level = level->getParent();
  }

  // Adds an entry into the position list, linking the given point in the
  // source code to the current position in the virtual machine code.  This is
  // used to print positions at runtime.
  void markPos(position pos);

  // When translation of the function is finished, this ties up loose ends
  // and returns the lambda.
  vm::lambda *close();

  // Finishes translating the initializer of a record.
  void closeRecord();

private: // Non-copyable
  void operator=(const coder&);
};

} // namespace trans

#endif
