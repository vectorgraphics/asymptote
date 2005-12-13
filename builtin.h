/*****
 * builtin.h
 * Tom Prince 2004/08/25
 *
 * Initialize builtins.
 *****/
#ifndef BUILTIN_H
#define BUILTIN_H

#include "vm.h"
#include "types.h"

namespace trans {

class tenv;
class venv;
class menv;

// The base environments for built-in types and functions
void base_tenv(tenv &);
void base_venv(venv &);
void base_menv(menv &);

// Add a function with one or more default arguments.
void addFunc(venv &ve, vm::bltin f, types::ty *result, const char *name, 
	     types::ty *t1=0, const char *s1="", bool d1=false,
	     types::ty *t2=0, const char *s2="", bool d2=false,
	     types::ty *t3=0, const char *s3="", bool d3=false,
	     types::ty *t4=0, const char *s4="", bool d4=false,
	     types::ty *t5=0, const char *s5="", bool d5=false,
	     types::ty *t6=0, const char *s6="", bool d6=false,
	     types::ty *t7=0, const char *s7="", bool d7=false,
	     types::ty *t8=0, const char *s8="", bool d8=false);
  
} //namespace trans

#endif //BUILTIN_H
