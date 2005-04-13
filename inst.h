/*****
 * inst.h
 * Andy Hammerlindl 2002/06/27
 * 
 * Descibes the items and instructions that are used by the virtual machine.
 *****/

#ifndef INST_H
#define INST_H

#include <string>
#include <vector>
#include <deque>
#include <iterator>
#include <iostream>

#include "errormsg.h"
#include "pool.h"
#include "item.h"

namespace vm {

// Forward declarations
struct inst; class stack;
 
// Manipulates the stack.
typedef void (*bltin)(stack *s);

class program
{
public:
  class label;
  program();
  inline void encode(inst i);
  inline void prepend(inst i);
  label begin();
  label end();
private:
  friend class label;
  class code_t : public std::deque<inst>, public memory::managed<code_t> {};
  code_t *code;
};

class program::label
{
public: // interface
  label() : where(0), code() {};
public: //interface
  void operator++();
  bool operator==(const label& right) const;
  bool operator!=(const label& right) const;
  inst& operator*() const;
  inst* operator->() const;
  friend ptrdiff_t offset(const label& left,
                          const label& right);
private:
  label (size_t where, code_t* code)
    : where(where), code(code) {};
  size_t where;
  code_t* code;
  friend class program;
};
  
// A function "lambda," that is, the code that runs a function.
// It also need the closure of the enclosing module or function to run.
struct lambda : public memory::managed<lambda> {
  // The instructions to follow.
  program code;

  // How many item can be pushed on the stack during the execution
  // of this function.
  int maxStackSize;

  // The number of parameters of the function.  This does not include the
  // closure of the enclosing module or function.
  int params;

  // The total number of items that will be stored in the closure of this
  // function.  Includes the higher closure, the parameters, and the local
  // variables.
  // NOTE: In order to help garbage collection, this could be modified to
  // have one array store escaping items, and another to store non-
  // escaping items.
  int vars;

  virtual ~lambda() {}
};

struct callable : public memory::managed<callable>
{
  virtual void call(stack *) = 0;
  virtual ~callable();
  virtual bool compare(callable*) { return false; }
};

class nullfunc : public callable 
{
private:
  nullfunc() {}
  static nullfunc func;
public:
  virtual void call (stack*);
  virtual bool compare(callable*);
  static callable* instance() { return &func; }
};

// How a function reference to a non-builtin function is stored.
struct func : public callable {
  lambda *body;
  frame closure;
  func () : body(0), closure() {};
  virtual void call (stack*);
  virtual bool compare(callable*);
};

class bfunc : public callable 
{
public:
  bfunc(bltin b) : func(b) {};
  virtual void call (stack *s) { func(s); }
  virtual bool compare(callable*);
private:
  bltin func;
};

class thunk : public callable
{
public:
  thunk(callable *f, item i) : func(f), arg(i) {};
  virtual void call (stack*);
private:
  callable *func;
  item arg;
};
  
// The code run is just a string of instructions.  The ops are actual commands
// to be run, but constants, labels, and other objects can be in the code.
struct inst {
  enum opcode {
    pop, intpush, constpush,
    varpush, varsave, fieldpush, fieldsave,
    builtin, jmp, cjmp, njmp, popcall,
    pushclosure, makefunc, ret
  };
  opcode op;
  position pos;
  union {
    int val;
    std::string *s;
    
    lambda *lfunc;
  };
  program::label label;
  bltin bfunc;
  item ref;
};

// Arrays are vectors with a push func for running in asymptote.
class array : public std::vector<item>, public memory::managed<array> {
public:
  array(size_t n)
    : std::vector<item>(n)
  {}

  void push(item i)
  {
    std::vector<item>::push_back(i);
  }

  template <typename T>
  T read(size_t i)
  {
    return get<T>((*this)[i]);
  }
};

template <typename T>
inline T read(array *a, size_t i)
{
  return a->array::read<T>(i);
}

// Prints one instruction (including arguments) and returns how many
// positions in the code stream were shown.
void printInst(std::ostream& out, program::label code, program::label base);

// Prints code until a ret opcode is printed.
void print(std::ostream& out, program base);

// Inline forwarding functions for vm::program
inline program::program()
  : code(new code_t) {}
inline program::label program::end()
{ return label(code->size(), code); }
inline program::label program::begin()
{ return label(0, code); }
inline void program::encode(inst i)
{ code->push_back(i); }
inline void program::prepend(inst i)
{ code->push_front(i); }
inline void program::label::operator++()
{ ++where; }
inline bool program::label::operator==(const label& right) const
{ return (code == right.code) && (where == right.where); }
inline bool program::label::operator!=(const label& right) const
{ return !(*this == right); }
inline inst& program::label::operator*() const
{ return (*code)[where]; }
inline inst* program::label::operator->() const
{ return &**this; }
inline ptrdiff_t offset(const program::label& left, const program::label& right)
{ return left.where - right.where; }

} // namespace vm

#endif
  
