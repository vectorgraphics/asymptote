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

#include <iostream>

using std::cout;
using std::endl;

#include "memory.h"
#include "frame.h"
#include "table.h"
#include "types.h"
#include "modifier.h"

using sym::symbol;
using types::ty;
using types::signature;

// Forward declaration.
namespace types {
  class record;
}
using types::record;

namespace trans {

// The type environment.
class tenv : public sym::table<ty *>
{};

class varEntry : public gc {
  ty *t;
  access *location;

  permission perm;
  record *r;  // The record the variable belongs to in the environment, ignores
              // static and dynamic qualifiers.

public:
  varEntry(ty *t, access *location)
    : t(t), location(location), perm(PUBLIC), r(0) {}

  varEntry(ty *t, access *location, permission perm, record *r)
    : t(t), location(location), perm(perm), r(r) {}

  ty *getType()
    { return t; }

  signature *getSignature()
  {
    return t->getSignature();
  }

  access *getLocation()
    { return location; }

  permission getPermission()
    { return perm; }

  record *getRecord()
    { return r; }

  void varEntry::checkPerm(action act, position pos, coder &c);

  // Encodes the access, but also checks permissions.
  void encode(action act, position pos, coder &c);
  void encode(action act, position pos, coder &c, frame *top);
};

#ifdef NOHASH //{{{
class venv : public sym::table<varEntry*> {
public:
  venv();

#if 0
  // Look for a function that exactly matches the signature given.
  varEntry *lookExact(symbol *name, signature *key);
#endif

  // Look for a function that exactly matches the type given.
  varEntry *lookByType(symbol *name, ty *t);

  // Checks if a function was added in the top scope as two identical
  // functions cannot be defined in one scope.
  varEntry *lookInTopScope(symbol *name, ty *t);

  // Return the type of the variable, if name is overloaded, return an
  // overloaded type.
  ty *getType(symbol *name);

  friend std::ostream& operator<< (std::ostream& out, const venv& ve);
  
  void list();
};

//}}}
#else //{{{
#define SHADOWING 1

// venv implemented with a hash table.  Will replace venv soon...
class venv {
public:
  struct key : public gc {
    symbol *name;
    ty *t;

    key(symbol *name, ty *t)
      : name(name), t(t) {}

    key(symbol *name, varEntry *v)
      : name(name), t(v->getType()) {}
  };
  struct value : public gc {
    varEntry *v;
    bool shadowed;
    value *next;  // The entry (of the same key) that this one shadows.

    value(varEntry *v)
      : v(v), shadowed(false), next(0) {}
  };
  struct namehash {
    size_t operator()(const symbol *name) const {
      return (size_t)name;
    }
  };
  struct nameeq {
    bool operator()(const symbol *s, const symbol *t) const {
      return s==t;
    }
  };
  struct keyhash {
    size_t hashSig(ty *t) const {
      signature *sig=t->getSignature();
      return sig ? sig->hash() : 0;
    }
    size_t operator()(const key k) const {
      return (size_t)(k.name) * 107 +
             (k.name->special ? k.t->hash() : hashSig(k.t));
    }
  };
  struct keyeq {
#define TEST_COLLISION 0
#if TEST_COLLISION
    bool base(const key k, const key l) const {
      return k.name==l.name &&
             (k.name->special ? equivalent(k.t, l.t) :
                                equivalent(k.t->getSignature(),
                                           l.t->getSignature()));
    }
    bool operator()(const key k, const key l) const;
#else
    bool operator()(const key k, const key l) const; 
#endif
  };


  // A hash table used to quickly look up a variable once its name and type are
  // known.  Includes all scopes.
  typedef mem::hash_map<key, value *, keyhash, keyeq> keymap;
  keymap all;

  // Similar hashes, one for each scope level.
  typedef mem::hash_multimap<key, value *, keyhash, keyeq> keymultimap;
  typedef mem::stack<keymultimap> mapstack;
  mapstack scopes;

  // A hash table indexed solely on the name, storing for each name the list of
  // all values of that name.  Used to get the (possibly overloaded) type of the
  // name.
  typedef mem::list<value *> values;
  typedef mem::hash_map<symbol *, values, namehash, nameeq> namemap;
  namemap names;

  void listValues(symbol *name, values &vals);

  // Helper function for endScope.
  void remove(key k);

public:
  venv() {
    beginScope();
  }

  void enter(symbol *name, varEntry *v);

  bool lookInTopScope(key k) {
    return scopes.top().find(k)!=scopes.top().end();
  }

  // Look for a function that exactly matches the type given.
  bool lookInTopScope(symbol *name, ty *t) {
    return lookInTopScope(key(name, t));
  }

  varEntry * lookByType(key k) {
    keymap::const_iterator p=all.find(k);
    return p!=all.end() ? p->second->v : 0;
  }
  
  // Look for a function that exactly matches the type given.
  varEntry *lookByType(symbol *name, ty *t) {
    return lookByType(key(name, t));
  }

  ty *getType(symbol *name);

  void beginScope() {
    scopes.push(keymultimap());
  }
  void endScope() {
    keymultimap &scope=scopes.top();
    for (keymultimap::iterator p=scope.begin(); p!=scope.end(); ++p) {
      remove(p->first);
    }
    scopes.pop();
  }

  // Prints a list of the variables to the standard output.
  void list();
};
#endif

} // namespace trans

#endif //ENTRY_H
