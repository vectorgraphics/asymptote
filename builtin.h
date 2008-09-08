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

extern const types::formal noformal;
  
// Add a function with one or more default arguments.
void addFunc(venv &ve, vm::bltin f, types::ty *result, const char *name, 
	     types::formal f1=noformal, types::formal f2=noformal,
	     types::formal f3=noformal, types::formal f4=noformal,
	     types::formal f5=noformal, types::formal f6=noformal,
	     types::formal f7=noformal, types::formal f8=noformal,
	     types::formal f9=noformal, types::formal fA=noformal,
	     types::formal fB=noformal, types::formal fC=noformal,
	     types::formal fD=noformal);
  
// Adds standard functions for a newly added types.
void addArrayOps(venv &ve, types::array *t);
void addRecordOps(venv &ve, types::record *r);
void addFunctionOps(venv &ve, types::function *f);

types::record *getGSLModule();
  
} //namespace trans

namespace run {
  void arrayDeleteHelper(vm::stack *Stack);
}

#endif //BUILTIN_H
