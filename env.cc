/*****
 * env.h
 * Andy Hammerlindl 2002/6/20
 *
 * Keeps track of the namespaces of variables and types when traversing
 * the abstract syntax.
 *****/

#include "env.h"
#include "genv.h"

using namespace types;

namespace trans {

env::env(genv &ge)
  : ge(ge), te(ge.te), ve(ge.ve), me(ge.me) {}

// Instances of this class are passed to types::ty objects so that they can call
// back to env when checking casting of subtypes.
class envCaster : public caster {
  env &e;
  symbol *name;
public:
  envCaster(env &e, symbol *name)
    : e(e), name(name) {}

  access *operator() (ty *target, ty *source) {
    return e.lookupCast(target, source, name);
  }

  bool castable(ty *target, ty *source) {
    return e.castable(target, source, name);
  }
};
  
access *env::baseLookupCast(ty *target, ty *source, symbol *name) {
  static identAccess id;

  assert(target->kind != ty_overloaded &&
         source->kind != ty_overloaded);

  // If errors already exist, don't report more.  This may, however, cause
  // problems with resoving the signature of an overloaded function.  The
  // abstract syntax should check if any of the parameters had an error before
  // finding the signature.
  if (target->kind == ty_error || source->kind == ty_error)
    return &id;
  else if (equivalent(target,source))
    return &id;
  else {
    varEntry *v=lookupVarByType(name,new function(target,source));
    return v ? v->getLocation() : 0;
  }
}

access *env::lookupCast(ty *target, ty *source, symbol *name) {
  access *a=baseLookupCast(target, source, name);
  if (a)
    return a;

  envCaster ec(*this, name);
  return source->castTo(target, ec);
}

bool env::castable(ty *target, ty *source, symbol *name) {
  struct castTester : public tester {
    env &e;
    symbol *name;

    castTester(env &e, symbol *name)
      : e(e), name(name) {}

    bool base(ty *t, ty *s) {
      access *a=e.baseLookupCast(t, s, name);
      if (a)
        return true;

      envCaster ec(e, name);
      return s->castable(t, ec);
    }
  };

  castTester ct(*this, name);
  return ct.test(target,source);
}

ty *env::castTarget(ty *target, ty *source, symbol *name) {
  struct resolver : public collector {
    env &e;
    symbol *name;

    resolver(env &e, symbol *name)
      : e(e), name(name) {}

    types::ty *base(types::ty *target, types::ty *source) {
      return e.castable(target, source, name) ? target : 0;
    }
  };
          
  resolver r(*this, name);
  return r.collect(target, source);
} 

ty *env::castSource(ty *target, ty *source, symbol *name) {
  struct resolver : public collector {
    env &e;
    symbol *name;

    resolver(env &e, symbol *name)
      : e(e), name(name) {}

    types::ty *base(types::ty *target, types::ty *source) {
      return e.castable(target, source, name) ? source : 0;
    }
  };
          
  resolver r(*this, name);
  return r.collect(target, source);
} 

record *env::getModule(symbol *id)
{
  record *m = ge.getModule(id);
  if (m) {
    return m;
  }
  else {
    return ge.loadModule(id);
  }
}

}
