/*****
 * entry.h
 * Andy Hammerlindl 2002/08/29
 *
 * All variables, built-in functions and user-defined functions reside
 * within the same namespace.  To keep track of all these, a table of
 * "entries" is used.
 *****/

#ifndef ENTRY_H
#define ENTRY_H

#include <vector>

#include "pool.h"
#include "frame.h"
#include "table.h"
#include "types.h"

using std::vector; 

using sym::symbol;
using types::ty;
using types::signature;

namespace trans {

// The type environment.
class tenv : public sym::table<ty *>
{};

class varEntry : public memory::managed<varEntry> {
  ty *t;
  access *location;

public:
  varEntry(ty *t, access *location)
    : t(t), location(location) {}

  ty *getType()
    { return t; }

  signature *getSignature()
  {
    return t->getSignature();
  }

  access *getLocation()
    { return location; }
};

class venv : public sym::table<varEntry*> {
public:
  venv();

  // Look for a function that exactly matches the signature given.
  varEntry *lookExact(symbol *name, signature *key);

  // Checks if a function was added in the top scope as two identical
  // functions cannot be defined in one scope.
  varEntry *lookInTopScope(symbol *name, signature *key);

  // Return the type of the variable, if name is overloaded, return an
  // overloaded type.
  ty *getType(symbol *name);

  friend std::ostream& operator<< (std::ostream& out, const venv& ve);
};

} // namespace trans

#endif //ENTRY_H
