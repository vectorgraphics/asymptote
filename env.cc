/*****
 * env.h
 * Andy Hammerlindl 2002/6/20
 *
 * Keeps track of the namespaces of variables and types when traversing
 * the abstract syntax.
 *****/

#include "env.h"
#include "genv.h"

namespace trans {

env::env(genv &ge)
  : ge(ge), te(ge.te), ve(ge.ve), me(ge.me) {}

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
