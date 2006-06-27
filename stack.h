/*****
 * stack.h
 * Andy Hammerlindl 2002/06/27
 * 
 * The general stack machine  used to run compiled camp code.
 *****/

#ifndef STACK_H
#define STACK_H

#include <iostream>
#include <deque>

#include "errormsg.h"
#include "vm.h"
#include "item.h"

namespace vm {

struct func;
class program;
class lambda;
class importInitMap;

class stack {
public:
  typedef frame* vars_t;

  struct importInitMap {
    virtual ~importInitMap() {}
    virtual lambda *operator[](mem::string) = 0;
  };

private:
  // stack for operands
  typedef mem::deque<item> stack_t;
  stack_t theStack;

  vars_t make_frame(size_t, vars_t closure);

  void draw(ostream& out);

  // Move arguments from stack to frame.
  void marshall(size_t args, vars_t vars);

  // The initializer functions for imports, indexed by name.
  importInitMap *initMap;

  // The stack stores a map of initialized imported modules by name, so that
  // each module is initialized only once and each import refers to the same
  // instance.
  typedef mem::map<CONST mem::string,frame *> importInstanceMap;
  importInstanceMap instMap;
  
  // Debugger variables:
  int debugOp; // 0=none, 1=step, 2=next.
  bool indebugger;
  position lastPos, breakPos;
  
public:
  stack() : debugOp(0), indebugger(false) {};
  
  ~stack() {};

  void setInitMap(importInitMap *i) {
    initMap=i;
  }

  // Runs the instruction listed in code, with vars as frame of variables.
  void run(program *code, vars_t vars);

  // Executes a function on top of the stack.
  void run(func *f);

  void breakpoint();
  void debug();
  
  // Put an import (indexed by name) on top of the stack, initializing it if
  // necessary.
  void load(mem::string index);

  // These are so that built-in functions can easily manipulate the stack
  void push(item next) {
    theStack.push_back(next);
  }
  template <typename T>
  void push(T next) {
    push((item)next);
  }
  item top() {
    return theStack.back();
  }
  item pop() {
    item ret = theStack.back();
    theStack.pop_back();
    return ret;
  }
  template <typename T>
  T pop()
  {
    return get<T>(pop());
  }
};

inline item pop(stack* s)
{
  return s->pop();
}

template <typename T>
inline T pop(stack* s)
{
  return get<T>(pop(s));
}
  
template <typename T>
inline T pop(stack* s, T defval)
{
  item it=pop(s);
  return isdefault(it) ? defval : get<T>(it);
}
  
class interactiveStack : public stack {
  vars_t globals;
public:
  interactiveStack();
    
  // Run a codelet, a small piece of code that uses globals as its frame.
  void run(lambda *codelet);
};

} // namespace vm

#endif // STACK_H
  
