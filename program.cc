/*****
 * program.cc
 * Tom Prince
 * 
 * The list of instructions used by the virtual machine.
 *****/

#include <iostream>
#include "program.h"


namespace vm {

static const char* opnames[] = {
  "pop", "intpush", "constpush", 
  "varpush", "varsave", "fieldpush", "fieldsave",
  "builtin", "jmp", "cjmp", "njmp", "call",
  "pushclosure", "makefunc", "ret",
  "alloc", "pushframe", "popframe"
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
    {
      out << " " << get<Int>(*code);
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

} // namespace vm
