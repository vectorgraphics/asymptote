/*****
 * import.cc
 * Andy Hammerlindl 2003/11/09
 *
 * Defines the import class, for the import of a module into a scope,
 * and the module environment for keeping track of imports.
 *****/

#include "import.h"
#include "importaccess.h"
#include "util.h"

namespace trans {

ty *menv::lookupType(symbol *s, scope_t &scope)
{
  // The possible answers in this scope.
  types::overloaded set;
 
  for(scope_iterator p = scope.begin();
      p != scope.end() && (*p).second;//XXXX
      ++p) {
    import *i = p->second;
    if (i == look(p->first)) {
      ty *t = i->m->lookupType(s);
      if (t)
	set.add(t);
    }
  }

  return set.simplify();
}

import *menv::lookupTypeImport(symbol *s, scope_t &scope)
{
  // Find the importer, if there are multiple matches, we have a
  // problem.  Ideally, the previous call to lookupType will ensure that
  // only valid allocations get to this stage.
  int matches = 0;
  import *holder = 0;
  
  for(scope_iterator p = scope.begin();
      p != scope.end() && (*p).second;//XXXX
      ++p) {
    import *i = p->second;
    if (i == look(p->first)) {
      ty *t = i->m->lookupType(s);
      if (t) {
	++matches;
	holder = i;
      }
    }
  }

  if (matches == 1)
    return holder;
  else 
    return 0;
}

#if 0
varEntry *menv::lookupExactVar(symbol *s, signature *sig, scope_t &scope)
{
  // If there is more than one exact match, then the reference to this variable
  // is ambiguous.  For instance, if module_a.func(int) and module_b.func(int)
  // are defined:
  //   import module_a;
  //   import module_b;
  //   int x = func(5);
  //
  // will give an error as func is ambiguous.  If module_a and module_b
  // are imported at different scopes, however, then the one in the
  // highest scope is used.
  types::overloaded set;

  varEntry *lastVar = 0;
  import   *lastImport = 0;
  
  // Find first applicable function.
  for(scope_iterator p = scope.begin();
      p != scope.end() && (*p).second;//XXXX
      ++p) {
    import *i = p->second;
    if (i == look(p->first)) {
      varEntry *v = i->m->lookupExactVar(s, sig);
      if (v) {
	set.add(v->getType());
	lastVar = v;
	lastImport = i;
      }
    }
  }
  
  ty *ret = set.simplify();
  if (ret) {
    if (ret->kind == types::ty_overloaded) {
      varEntry *v = new varEntry(ret, 0);
      return v;
    }
    else
      return importedVarEntry(lastVar, lastImport);
  }
  else
    return 0;
}
#endif

varEntry *menv::lookupVarByType(symbol *s, ty *t, scope_t &scope)
{
  types::overloaded set;

  varEntry *lastVar = 0;
  import   *lastImport = 0;
  
  // Find first applicable function.
  for(scope_iterator p = scope.begin();
      p != scope.end() && (*p).second;//XXXX
      ++p) {
    import *i = p->second;
    if (i == look(p->first)) {
      varEntry *v = i->m->lookupVarByType(s, t);
      if (v) {
	set.add(v->getType());
	lastVar = v;
	lastImport = i;
      }
    }
  }
  
  ty *ret = set.simplify();
  if (ret) {
    if (ret->kind == types::ty_overloaded) {
      varEntry *v = new varEntry(ret, 0);
      return v;
    }
    else
      return importedVarEntry(lastVar, lastImport);
  }
  else
    return 0;
}

ty *menv::varGetType(symbol *s, scope_t& scope)
{
  // The possible answers in this scope.
  types::overloaded set;
 
  for(scope_iterator p = scope.begin();
      p != scope.end() && (*p).second;//XXXX
      ++p) {
    import *i = p->second;
    if (i == look(p->first)) {
      ty *t = i->m->varGetType(s);
      if (t)
        set.add(t);
    }
  }

  return set.simplify();
}

varEntry *menv::importedVarEntry(varEntry *ent, import *i)
{
  access *a = new importAccess(i->getLocation(),
                               i->getModule()->getLevel(),
			       ent->getLocation());
  
  varEntry *ient = new varEntry(ent->getType(), a);

  return ient;
}
  
}
