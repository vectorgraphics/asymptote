/*****
 * stack.h
 * Andy Hammerlindl 2002/06/27
 * 
 * The general stack machine that will be used to run compiled camp
 * code.
 *****/

#ifndef STACK_H
#define STACK_H

#include <iostream>
#include <string>
#include <deque>
#include <stack>

#include "errormsg.h"
#include "inst.h"

namespace vm {

class stack {
public:
  typedef frame vars_t;

private:
  // stack for operands
  typedef std::deque<item> stack_t;
  stack_t theStack;

  // array for global variables */
  int numGlobals;
  vars_t globals;

  lambda *body;

  vars_t vars;
  vars_t make_frame(size_t);

  program::label ip;

  void draw(ostream& out);
public:
  stack(int numGlobals);
  ~stack();

  // Executes a function on top of the stack.
  void run(func *f);
  void run(lambda *l);

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

  // Returns the position of the stack when the running lambda has line 
  // number information included.
  position getPos();
};

template <typename T>
T pop(stack* s)
{
  return s->pop<T>();
}

inline void error(stack *s, const char* message)
{
  em->runtime(s->getPos());
  *em << message;
  em->sync();
  throw handled_error();
}
  
} // namespace vm

#endif
  
