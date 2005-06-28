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
#include "import.h"
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

  // These tables keep track of type, variable definitions, and of
  // imported modules.
  tenv &te;
  venv &ve;
  menv &me;

  access *baseLookupCast(ty *target, ty *source, symbol *name);

public:
  // Start an environment for a file-level module.
  env(genv &ge);

  env(const env&);
  
  void beginScope()
  {
    te.beginScope(); ve.beginScope(); me.beginScope();
  }
  void endScope()
  {
    te.endScope(); ve.endScope(); me.endScope();
  }

  ty *lookupType(symbol *s)
  {
    // Search in local types.
    ty *t = te.look(s);
    if (t)
      return t;

    // Search in modules.
    t = me.lookupType(s);
    if (t)
      return t;
    
    // Search module names.
    import *i = me.look(s);
    if (i)
      return i->getModule();
    
    // No luck.
    return 0;
  }

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

#if 0
  varEntry *lookupExactVar(symbol *name, signature *sig)
  {
    // Search in local vars.
    varEntry *v = ve.lookExact(name, sig);
    if (v)
      return v;

    // Search in modules.
    v = me.lookupExactVar(name, sig);
    if (v)
      return v;
    
    // Search module name.
    import *i = me.look(name);
    if (i)
      return i->getVarEntry();

    // No luck.
    return 0;
  }
#endif

  varEntry *lookupVarByType(symbol *name, ty *t)
  {
    // Search in local vars.
    varEntry *v = ve.lookByType(name, t);
    if (v)
      return v;

    // Search in modules.
    v = me.lookupVarByType(name, t);
    if (v)
      return v;
    
    // Search module name.
    import *i = me.look(name);
    if (i)
      return i->getVarEntry();

    // No luck.
    return 0;
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
    // NOTE: This overhead seems unnecessarily slow.
    types::overloaded o;
    
    ty *t = ve.getType(name);
    if (t)
      o.add(t);

    t = me.varGetType(name);
    if (t)
      o.addDistinct(t);

    import *i = me.look(name);
    if (i)
      o.addDistinct(i->getModule());

    return o.simplify();
  }

  void addType(position pos, symbol *name, ty *desc)
  {
    if (te.lookInTopScope(name)) {
      em->error(pos);
      *em <<  "type \'" << *name << "\' previously declared";
    }
    te.enter(name, desc);
  }
  
  void addVar(position pos, symbol *name, varEntry *desc, bool ignore=false)
  {
    // For now don't check for multiple variables, as this makes adding casts
    // and initializers harder.  Figure out what to do about this.
#if 0
    signature *sig = desc->getSignature();
    if (ve.lookInTopScope(name, sig)) {
      if(ignore) return;
      em->error(pos);
      if (sig)
        *em << "function variable \'" << *name << *sig
            << "\' previously declared";
      else
        *em << "variable '" << *name <<  "' previously declared";
    }
#endif
    ve.enter(name, desc);
  }

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

  record *getModule(symbol *id);

private: // Non-copyable
  void operator=(const env&);
};

} // namespace trans

#endif
