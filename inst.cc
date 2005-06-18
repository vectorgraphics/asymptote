/*****
 * stack.cc
 * Andy Hammerlindl 2002/06/27
 *
 * The general stack machine that will be used to run compiled camp
 * code.
 *****/

#include <cassert>
#include <cstdio>
#include <iostream>
#include <iterator>

#include "stack.h"
#include "runtime.h"

namespace vm {

static const char* opnames[] = {
  "pop", "intpush", "constpush", 
  "varpush", "varsave", "fieldpush", "fieldsave",
  "builtin", "jmp", "cjmp", "njmp", "call",
  "pushclosure", "makefunc", "ret",
  "alloc"
};
static const int numOps = (int)(sizeof(opnames)/sizeof(char *));

void printInst(ostream& out, const program::label& code,
	       const program::label& base)
{
  out.width(4);
  out << offset(base,code) << " ";
  
  int i = (int)code->op;
  
  if (i < 0 || i >= numOps) {
    out << "<<invalid op>> " << i;
  }
  out << opnames[i];

  switch (code->op) {
    case inst::intpush:
    case inst::varpush:
    case inst::varsave:
    case inst::fieldpush:
    case inst::fieldsave:
    case inst::alloc:
    {
      out << " " << get<int>(*code);
      break;
    }

    case inst::builtin:
    {      
      out << " " << get<bltin>(*code) << " ";
      break;
    }

    case inst::jmp:
    case inst::cjmp:
    case inst::njmp:
    {
      char f = out.fill('0');
      out << " i";
      out.width(4);
      out << offset(base,get<program::label>(*code));
      out.fill(f);
      break;
    }

    case inst::makefunc:
    {
      out << " " << get<lambda*>(*code) << " ";
      break;
    }
    
    default: {
      /* nothing else to do */
      break;
    }
  };
}

void print(ostream& out, program *base)
{
  program::label code = base->begin();
  bool active = true;
  while (active) {
    if (code->op == inst::ret || 
        code->op < 0 || code->op >= numOps)
      active = false;
    printInst(out, code, base->begin());
    out << '\n';
    ++code;
  }
}

callable::~callable()
{}

void func::call(stack *s)
{
  s->run(this);
}

bool func::compare(callable* F)
{
  if (func* f=dynamic_cast<func*>(F))
    return (body == f->body) && (closure == f->closure);
  else return false;
}

bool bfunc::compare(callable* F)
{
  if (bfunc* f=dynamic_cast<bfunc*>(F))
    return (func == f->func);
  else return false;
}

void thunk::call(stack *s)
{
  s->push(arg);
  func->call(s);
}

nullfunc nullfunc::func;
void nullfunc::call(stack *)
{
  error("dereference of null function");
}

bool nullfunc::compare(callable* f)
{
  return f == &func;
}

} // namespace vm
