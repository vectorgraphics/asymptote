/*****
 * program.cc
 * Tom Prince
 * 
 * The list of instructions used by the virtual machine.
 *****/

#include <iostream>
#include "util.h"
#include "callable.h"
#include "program.h"


namespace vm {

static const char* opnames[] = {
  "pop", "intpush", "constpush", 
  "varpush", "varsave", "fieldpush", "fieldsave",
  "builtin", "jmp", "cjmp", "njmp", "call",
  "pushclosure", "makefunc", "ret",
  "alloc", "pushframe", "popframe",

  "push_default",
  "jump_if_not_default",

#ifdef COMBO
  "varpop", "fieldpop",
  "gejmp"
#endif
};
static const Int numOps = (Int)(sizeof(opnames)/sizeof(char *));

#ifdef DEBUG_BLTIN
mem::map<bltin,string> bltinRegistry;

void registerBltin(bltin b, string s) {
  bltinRegistry[b] = s;
}
string lookupBltin(bltin b) {
  return bltinRegistry[b];
}
#endif


ostream& operator<< (ostream& out, const item& i)
{
#if COMPACT
  out << "<item>";
#else
  // TODO: Make a data structure mapping typeids to print functions.
  if (i.empty())
    out << "empty";
  else if (isdefault(i))
    out << "default";
  else if (i.type() == typeid(Int))
    out << "Int, value = " << get<Int>(i);
  else if (i.type() == typeid(double))
    out << "real, value = " << get<double>(i);
  else if (i.type() == typeid(string))
    out << "string, value = " << get<string>(i);
  else if (i.type() == typeid(callable))
    out << *(get<callable *>(i));
  else if (i.type() == typeid(frame)) {
    out << "frame";
#ifdef DEBUG_FRAME
    out << " " << (get<frame *>(i))->getName();
#endif
  }
  else
    out << "type " << demangle(i.type().name());
#endif

  return out;
}

void printInst(ostream& out, const program::label& code,
               const program::label& base)
{
  out.width(4);
  out << offset(base,code) << " ";
  
  Int i = (Int)code->op;
  
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
#ifdef COMBO
    case inst::varpop:
    case inst::fieldpop:
#endif
    {
      out << " " << get<Int>(*code);
      break;
    }

    case inst::constpush:
    {
      item c = code->ref;
      out << " " << c;
      break;
    }

#ifdef DEBUG_BLTIN
    case inst::builtin:
    {
      string s=lookupBltin(get<bltin>(*code));
      out << " " << (!s.empty() ? s : "<unnamed>") << " ";
      break;
    }
#endif

    case inst::jmp:
    case inst::cjmp:
    case inst::njmp:
    case inst::jump_if_not_default:
#ifdef COMBO
    case inst::gejmp:
#endif
    {
      char f = out.fill('0');
      out << " i";
      out.width(4);
      out << offset(base,get<program::label>(*code));
      out.fill(f);
      break;
    }

#ifdef DEBUG_FRAME
    case inst::makefunc:
    {
      out << " " << get<lambda*>(*code)->name << " ";
      break;
    }
#endif
    
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

} // namespace vm
