/*****
 * entry.cc
 * Andy Hammerlindl 2002/08/29
 *
 * All variables, built-in functions and user-defined functions reside
 * within the same namespace.  To keep track of all these, table of
 * "entries" is used.
 *****/

#include <cmath>
#include <utility>
#include "entry.h"

// For function resolution debugging.
#define DFL 0

using types::signature;

namespace trans {

venv::venv()
{
}

varEntry *venv::lookExact(symbol *name, signature *key)
{
  // Find first applicable function.
  name_t &list = names[name];
  for(name_iterator p = list.begin();
      p != list.end();
      ++p) {
    if (equivalent((*p)->getSignature(), key))
      return *p;
  }
  return 0;
}

void venv::list()
{
  // List all functions.
  for(names_t::iterator N = names.begin(); N != names.end(); ++N) {
    symbol *s=N->first;
    name_t &list=names[s];
    for(name_iterator p = list.begin(); p != list.end(); ++p) {
      signature *sig=(*p)->getSignature();
      if(sig) {
	std::cout << *((types::function *) (*p)->getType())->getResult() << " "
		  << *s;
	std::cout << *sig << ";" << std::endl;
      }
    }
  }
}

varEntry *venv::lookInTopScope(symbol *name, signature *key)
{
  scope_t &scope = scopes.front();
  for (scope_iterator p = scope.lower_bound(name);
       p != scope.upper_bound(name);
       ++p) {
    if (name == p->first &&
        equivalent(p->second->getSignature(), key))//XXX
      return p->second;
  }
  return 0;
}

ty *venv::getType(symbol *name)
{
  types::overloaded set;

  // Find all applicable functions in scope.
  name_t &list = names[name];
  
  for(name_iterator p = list.begin();
      p != list.end();
      ++p) {
      set.addDistinct((*p)->getType());
  }

  return set.simplify();
}

} // namespace trans
