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

  lambda *body;

  vars_t make_frame(size_t, vars_t closure);

  program::label ip;

  void draw(ostream& out);

  // Move arguments from stack to frame.
  void marshall(int args, vars_t vars);
public:
  stack();
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

inline item pop(stack* s)
{
  return s->pop();
}

template <typename T>
inline T pop(stack* s)
{
  return get<T>(pop(s));
}

void error(stack *s, const char* message);
  
} // namespace vm

#endif
  
