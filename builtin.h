/*****
 * builtin.h
 * Tom Prince 2004/08/25
 *
 * Initialize builtins.
 *****/
#ifndef BUILTIN_H
#define BUILTIN_H

namespace vm {
class stack;
typedef void (*bltin)(stack*);
}
namespace types {
class ty;
}

namespace trans {

class tenv;
class venv;
class menv;

// The base environments for built-in types and functions
void base_tenv(tenv &);
void base_venv(venv &);
void base_menv(menv &);

void addFunc(venv &ve, vm::bltin f, types::ty *result, const char *name, 
             types::ty *t1 = 0, types::ty *t2 = 0, types::ty *t3 = 0,
             types::ty *t4 = 0, types::ty *t5 = 0, types::ty *t6 = 0,
             types::ty *t7 = 0, types::ty *t8 = 0);

} //namespace trans

#endif //BUILTIN_H
