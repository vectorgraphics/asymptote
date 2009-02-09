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

#include "common.h"
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

// An entry is associated to a name in the (variable or type) environment, and
// has permission based on the enclosing records where it was defined or
// imported.
class entry : public gc {
  struct pr {
    permission perm;
    record *r;

    pr(permission perm, record *r)
      : perm(perm), r(r) {}

    // Returns true if the permission allows access in this context.
    bool check(action act, coder &c);

    // Reports an error if permission is not allowed.
    void report(action act, position pos, coder &c);
  };
  
  mem::list<pr> perms;

  void addPerm(permission perm, record *r) {
    // Only store restrictive permissions.
    if (perm != PUBLIC && r)
      perms.push_back(pr(perm,r));
  }

  // The record where the variable or type is defined, or 0 if the entry is
  // not a field.
  record *where;

  // The location (file and line number) where the entry was defined.
  position pos;
  
public:
  entry(record *where, position pos) : where(where), pos(pos) {}
  entry(permission perm, record *r, record *where, position pos)
    : where(where), pos(pos) {
    addPerm(perm, r);
  }

  // (Non-destructively) merges two entries, appending permission lists.
  // The 'where' member is taken from the second entry.
  entry(entry &e1, entry &e2);
  
  // Create an entry with one more permission in the list.
  entry(entry &base, permission perm, record *r);

  bool checkPerm(action act, coder &c);
  void reportPerm(action act, position pos, coder &c);

  record *whereDefined() {
    return where;
  }
  
  position getPos() {
    return pos;
  }
};
    
class varEntry : public entry {
  ty *t;
  access *location;

public:
  varEntry(ty *t, access *location, record *where, position pos)
    : entry(where, pos), t(t), location(location) {}

  varEntry(ty *t, access *location, permission perm, record *r,
           record *where, position pos)
    : entry(perm, r, where, pos), t(t), location(location) {}

  // (Non-destructively) merges two varEntries, creating a qualified varEntry.
  varEntry(varEntry &qv, varEntry &v);

  ty *getType()
  { return t; }

  signature *getSignature()
  {
    return t->getSignature();
  }

  access *getLocation()
  { return location; }

  frame *getLevel();

  // Encodes the access, but also checks permissions.
  void encode(action act, position pos, coder &c);
  void encode(action act, position pos, coder &c, frame *top);
};

varEntry *qualifyVarEntry(varEntry *qv, varEntry *v);

// As looked-up types can be allocated in a new expression, we need to know
// what frame they should be allocated on.  Type entries store this extra
// information along with the type.
class tyEntry : public entry {
public:
  ty *t;
  varEntry *v;  // NOTE: Name isn't very descriptive.

  tyEntry(ty *t, varEntry *v, record *where, position pos)
    : entry(where, pos), t(t), v(v) {}

  tyEntry(tyEntry *base, permission perm, record *r)
    : entry(*base, perm, r), t(base->t), v(base->v) {}

  // Records need a varEntry that refers back to the qualifier qv; i.e. in
  // the last new of the code
  //   struct A {
  //     struct B {}
  //   }
  //   A a=new A;
  //   unravel a;
  //   new B;
  // we need to put a's frame on the stack before allocating an instance of B.
  // NOTE: A possible optimization could be to only qualify the varEntry if
  // the type is a record, as other types don't use the varEntry.
private:
  tyEntry(tyEntry *base, varEntry *qv)
    : entry(*base, *qv), t(base->t), v(qualifyVarEntry(qv, base->v)) {}

public:
  // Since the constructor can only be used when qv is non-null it is private
  // for safety reasons, and we provide this method instead.
  friend tyEntry *qualifyTyEntry(varEntry *qv, tyEntry *ent);
};

inline tyEntry *qualifyTyEntry(varEntry *qv, tyEntry *ent) {
  return qv ? new tyEntry(ent, qv) : ent;
}

// The type environment.
class tenv : public sym::table<tyEntry *> {
  bool add(symbol *dest, names_t::value_type &x, varEntry *qualifier,
           coder &c);
public:
  // Add the entries in one environment to another, if qualifier is
  // non-null, it is a record and the source environment is its types.  The
  // coder is used to see which entries are accessible and should be added.
  void add(tenv& source, varEntry *qualifier, coder &c);

  // Adds entries of the name src in source as the name dest, returning true if
  // any were added.
  bool add(symbol *src, symbol *dest,
           tenv& source, varEntry *qualifier, coder &c);
};

#ifdef NOHASH //{{{
class venv : public sym::table<varEntry*> {
public:
  venv() {}

  // This is an optimization in the hashtable version that is duplicated here
  // for compatibility.  It is identical to venv().
  struct file_env_tag {};
  venv(file_env_tag) {}

#if 0
  // Look for a function that exactly matches the signature given.
  varEntry *lookExact(symbol *name, signature *key);
#endif

  // Add the entries in one environment to another, if qualifier is
  // non-null, it is a record and the source environment are its fields.
  // The coder is necessary to check which variables are accessible and
  // should be added.
  void add(venv& source, varEntry *qualifier, coder &c);

  // Add all unshadowed variables from source of the name src as variables
  // named dest.  Returns true if at least one was added.
  bool add(symbol *src, symbol *dest,
           venv& source, varEntry *qualifier, coder &c);

  // Look for a function that exactly matches the type given.
  varEntry *lookByType(symbol *name, ty *t);

  // Checks if a function was added in the top scope as two identical
  // functions cannot be defined in one scope.
  varEntry *lookInTopScope(symbol *name, ty *t);

  // Return the type of the variable, if name is overloaded, return an
  // overloaded type.
  ty *getType(symbol *name);

  friend std::ostream& operator<< (std::ostream& out, const venv& ve);
  
  // Prints a list of the variables to the standard output.
  void list(record *module=0);
};

//}}}
#else //{{{

// venv implemented with a hash table.
class venv {
  struct key : public gc {
    symbol *name;
    ty *t;

    key(symbol *name, ty *t)
      : name(name), t(t) {}

    key(symbol *name, varEntry *v)
      : name(name), t(v->getType()) {}
  };
  friend ostream& operator<< (ostream& out, const venv::key &k);

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
  typedef mem::unordered_map<key, value *, keyhash, keyeq> keymap;
  keymap all;

  // Similar hashes, one for each scope level.
  typedef mem::unordered_multimap<key, value *, keyhash, keyeq> keymultimap;
  typedef mem::stack<keymultimap> mapstack;
  mapstack scopes;

  // A hash table indexed solely on the name, storing for each name the list of
  // all values of that name.  Used to get the (possibly overloaded) type
  // of the name.
  typedef mem::list<value *> values;
  typedef mem::unordered_map<symbol *, values, namehash, nameeq> namemap;
  namemap names;

  void listValues(symbol *name, values &vals, record *module);

  // Helper function for endScope.
  void remove(key k);

  // These are roughly the size the hashtables will be after loading the
  // builtin functions and plain module.
  static const size_t fileAllSize=2000;
  static const size_t namesAllSize=1000;
public:
  venv() {
    beginScope();
  }

  // Most file level modules automatically import plain, so allocate hashtables
  // big enough to hold it in advance.
  struct file_env_tag {};
  venv(file_env_tag)
    : all(fileAllSize), names(namesAllSize)
  {
    beginScope();
  }

  void enter(symbol *name, varEntry *v);

  // Add the entries in one environment to another, if qualifier is
  // non-null, it is a record and entries of the source environment are its
  // fields.  The coder is necessary to check which variables are accessible and
  // should be added.
  void add(venv& source, varEntry *qualifier, coder &c);

  // Add all unshadowed variables from source of the name src as variables
  // named dest.  Returns true if at least one was added.
  bool add(symbol *src, symbol *dest,
           venv& source, varEntry *qualifier, coder &c);

  bool lookInTopScope(key k) {
    return scopes.top().find(k)!=scopes.top().end();
  }

  // Look for a function that exactly matches the type given.
  bool lookInTopScope(symbol *name, ty *t) {
    return lookInTopScope(key(name, t));
  }

  varEntry *lookByType(key k) {
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
  
  // Adds the definitions of the top-level scope to the level underneath,
  // and then removes the top scope.
  void collapseScope() {
    // NOTE: May be expensively copying a large hash table.
    keymultimap top=scopes.top();
    scopes.pop();

    keymultimap& underneath=scopes.top();
    underneath.insert(top.begin(), top.end());
  }

  // Prints a list of the variables to the standard output.
  void list(record *module=0);

  // Adds to l, all names prefixed by start.
  void completions(mem::list<symbol *>& l, string start);
};
#endif

} // namespace trans

#endif //ENTRY_H
