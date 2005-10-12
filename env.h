/*****
 * env.h
 * Andy Hammerlindl 2002/6/20
 *
 * Keeps track of the namespaces of variables and types when traversing
 * the abstract syntax.
 *****/

#ifndef ENV_H
#define ENV_H

#include <list>
#include <map>
#include <stack>

#include "errormsg.h"
#include "entry.h"
#include "builtin.h"
#include "types.h"
#include "record.h"
#include "util.h"


namespace trans {

using std::list;

using sym::symbol;
using types::ty;
using types::function;
using types::record;

class genv;

class env {
  // The global environment - keeps track of modules.
  genv &ge;

  // These tables keep track of type and variable definitions.
  tenv te;
  venv ve;

  access *baseLookupCast(ty *target, ty *source, symbol *name);

public:
  // Start an environment for a file-level module.
  env(genv &ge);

  env(const env&);
  
  void beginScope()
  {
    te.beginScope(); ve.beginScope();
  }
  void endScope()
  {
    te.endScope(); ve.endScope();
  }

  tyEntry *lookupTypeEntry(symbol *s)
  {
    return te.look(s);
  }

  ty *lookupType(symbol *s)
  {
    tyEntry *ent=lookupTypeEntry(s);
    return ent ? ent->t : 0;
  }

#if 0 //{{{
  // Returns the import in which the type is contained.
  import *lookupTypeImport(symbol *s)
  {
    // If the typename is in the local environment, it is not imported.
    if (te.look(s))
      return 0;

    // Search in modules.
    import *i = me.lookupTypeImport(s);
    if (i)
      return i;

    // Search the module name, if it is module, it is its own import?
    // NOTE: Types in this fashion should not be allocated!
    i = me.look(s);
    if (i)
      return i;

    // Error!
    assert(False);
    return 0;
  }
#endif //}}}

  varEntry *lookupVarByType(symbol *name, ty *t)
  {
    // Search in local vars.
    return ve.lookByType(name, t);
  }

  access *lookupInitializer(ty *t)
  {
    // The initializer's type is a function returning the desired type.
    function *it=new function(t);
    varEntry *v=lookupVarByType(symbol::initsym,it);

    // If not in the environment, try the type itself.
    return v ? v->getLocation() : t->initializer();
  }

  // Find the function that handles casting between the types.
  // The name is "cast" for implicitCasting and "ecast" for explicit (for now).
  access *lookupCast(ty *target, ty *source, symbol *name);
  bool castable(ty *target, ty *source, symbol *name);

  // Given overloaded types, this resolves which types should be the target and
  // the source of the cast.
  ty *castTarget(ty *target, ty *source, symbol *name);
  ty *castSource(ty *target, ty *source, symbol *name);

  ty *varGetType(symbol *name)
  {
    return ve.getType(name);
  }

  void addType(symbol *name, tyEntry *desc)
  {
    te.enter(name, desc);
  }
  
  void addVar(symbol *name, varEntry *desc)
  {
    // Don't check for multiple variables, as this makes adding casts
    // and initializers harder.
    ve.enter(name, desc);
  }

#if 0 //{{{
  void enterImport(symbol *name, import *i)
  {
    me.enter(name, i);
  }
    
  void addImport(position pos, symbol *name, import *i)
  {
    if (me.lookInTopScope(name)) {
      em->error(pos);
      *em << "multiple imports under name '" << *name << "'";
      return;
    }
    if(settings::verbose > 1)
      std::cerr << "Importing " <<  *name << std::endl;
    enterImport(name, i);
  }
#endif //}}}

  record *getModule(symbol *id, std::string filename);

private: // Non-copyable
  void operator=(const env&);
};

} // namespace trans

#endif
