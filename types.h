/*****
 * types.h
 * Andy Hammerlindl 2002/06/20
 *
 * Used by the compiler as a way to keep track of the type of a variable
 * or expression.
 *
 *****/

#ifndef TYPES_H
#define TYPES_H

#include <iostream>
#include <cstdio>
#include <cassert>
#include <vector>

#include "errormsg.h"
#include "symbol.h"
#include "memory.h"
#include "util.h"

using std::ostream;

using sym::symbol;

// Forward declaration.
namespace trans {
class access;
class varEntry;
}
namespace absyntax {
class varinit;
extern varinit *Default;
}

namespace types {

enum ty_kind {
  ty_void,
  ty_null,
  ty_record,	// "struct" in Asymptote language
  ty_function,
  ty_error,
  ty_overloaded,
 
  ty_boolean,	// "bool" in Asymptote language
  ty_int,
  ty_real,
  
  ty_string,
  ty_pair,
  ty_triple,
  ty_transform,
  ty_guide,
  ty_path,
  ty_pen,
  ty_picture,	// "frame" in Asymptote language
  ty_file,
  ty_code,
  ty_array
};

// Forward declarations.
class ty;
struct signature;

// Checks if two types are equal in the sense of the language.
// That is primitive types are equal if they are the same kind.
// Structures are equal if they come from the same struct definition.
// Arrays are equal if their cell types are equal.
bool equivalent(ty *t1, ty *t2);

// If special is true, this is the same as above.  If special is false, just the
// signatures are compared.
bool equivalent(ty *t1, ty *t2, bool special);

class caster {
public:
  virtual ~caster() {}
  virtual trans::access *operator() (ty *target, ty *source) = 0;
  virtual bool castable(ty *target, ty *source) = 0;
};

class ty : public gc {
public:
  const ty_kind kind;
  ty(ty_kind kind)
    : kind(kind) {}
  virtual ~ty();

  virtual void print (ostream& out) const;
  virtual void printVar (ostream& out, symbol *name) const {
    print(out);
    out << " " << *name;
  }


  // Returns true if the type is a user-defined type or the null type.
  // While the pair, path, etc. are stored by reference, this is
  // transparent to the user.
  virtual bool isReference() {
    return true;
  }

  virtual signature *getSignature() {
    return 0;
  }

  virtual bool primitive() {
    return false;
  }

  // If a default initializer is not stored in the environment, the abstract
  // syntax asks the type if it has a "default" default initializer, by calling
  // this method.
  virtual trans::access *initializer() {
    return 0;
  }

  // If a cast function is not stored in the environment, ask the type itself.
  // This handles null->record casting, and the like.  The caster is used as a 
  // callback to the environment for casts of subtypes.
  virtual trans::access *castTo(ty *, caster &) {
    return 0;
  }

  // Just checks if a cast is possible.
  virtual bool castable(ty *target, caster &c) {
    return castTo(target, c);
  }

  // For pair's x and y, and array's length, this is a special type of
  // "field".
  // In actually, it returns a function which takes the object as its
  // parameter and returns the necessary result.
  // These should not have public permission, as modifying them would
  // have strange results.
  virtual trans::varEntry *virtualField(symbol *, signature *) {
    return 0;
  }

  // varGetType for virtual fields.
  // Unless you are using functions for virtual fields, the base implementation
  // should work fine.
  virtual ty *virtualFieldGetType(symbol *id);

#if 0
  // Returns the type.  In case of functions, return the equivalent type
  // but with no default values for parameters.
  virtual ty *stripDefaults()
  {
    return this;
  }
#endif

  // Returns true if the other type is equivalent to this one.
  // The general function equivalent should be preferably used, as it properly
  // handles overloaded type comparisons.
  virtual bool equiv(ty *other)
  {
    return this==other;
  }


  // Returns a number for the type for use in a hash table.  Equivalent types
  // must yield the same number.
  virtual size_t hash() = 0;
};

class primitiveTy : public ty {
public:
  primitiveTy(ty_kind kind)
    : ty(kind) {}
  
  bool primitive() {
    return true;
  }

  bool isReference() {
    return false;
  }
  
  virtual trans::varEntry *virtualField(symbol *, signature *);

  bool equiv(ty *other)
  {
    return this->kind==other->kind;
  }

  size_t hash() {
    return (size_t)kind + 47;
  }
};

class nullTy : public primitiveTy {
public:
  nullTy()
    : primitiveTy(ty_null) {}
  
  bool isReference() {
    return true;
  }

  trans::access *castTo(ty *target, caster &);

  size_t hash() {
    return (size_t)kind + 47;
  }
};

// Ostream output, just defer to print.
inline ostream& operator<< (ostream& out, const ty& t)
  { t.print(out); return out; }

struct array : public ty {
  ty *celltype;
  ty *pushtype;
  ty *poptype;
  ty *appendtype;
  ty *inserttype;
  ty *deletetype;

  array(ty *celltype)
    : ty(ty_array), celltype(celltype), pushtype(0), poptype(0),
      appendtype(0), inserttype(0), deletetype(0) {}

  virtual bool isReference() {
    return true;
  }

  bool equiv(ty *other) {
    return other->kind==ty_array &&
           equivalent(this->celltype,((array *)other)->celltype);
  }

  size_t hash() {
    return 1007 * celltype->hash();
  }

  int depth() {
    if (array *cell=dynamic_cast<array *>(celltype))
      return cell->depth() + 1;
    else
      return 1;
  }

  void print(ostream& out) const
    { out << *celltype << "[]"; }

  ty *pushType();
  ty *popType();
  ty *appendType();
  ty *insertType();
  ty *deleteType();

  // Initialize to an empty array by default.
  trans::access *initializer();

  // NOTE: General vectorization of casts would be here.

  // Add length and push as virtual fields.
  ty *virtualFieldGetType(symbol *id);
  trans::varEntry *virtualField(symbol *id, signature *sig);
};

/* Base types */
ty *primVoid();
ty *primNull();
ty *primError();
ty *primBoolean();
ty *primInt();
ty *primReal();
ty *primString();
ty *primPair();
ty *primTriple();
ty *primTransform();
ty *primGuide();
ty *primPath();
ty *primPen();
ty *primPicture();
ty *primFile();
ty *primCode();
ty *primArray();
  
ty *boolArray();
ty *intArray();
ty *realArray();
ty *pairArray();
ty *tripleArray();
ty *stringArray();
ty *transformArray();
ty *pathArray();
ty *penArray();
ty *guideArray();

ty *boolArray2();
ty *intArray2();
ty *realArray2();
ty *pairArray2();
ty *tripleArray2();
ty *stringArray2();
ty *penArray2();

ty *boolArray3();
ty *intArray3();
ty *realArray3();
ty *pairArray3();
ty *tripleArray3();
ty *stringArray3();
  
struct formal {
  ty *t;
  symbol *name;
  absyntax::varinit *defval;
  bool Explicit;
  
  formal(ty *t,
         symbol *name=0,
         bool optional=false,
         bool Explicit=false)
    : t(t), name(name),
      defval(optional ? absyntax::Default : 0), Explicit(Explicit) {}

  formal(ty *t,
         const char *name,
         bool optional=false,
         bool Explicit=false)
    : t(t), name(symbol::trans(name)),
      defval(optional ? absyntax::Default : 0), Explicit(Explicit) {}

  friend ostream& operator<< (ostream& out, const formal& f);
};

bool equivalent(formal& f1, formal& f2);

typedef mem::vector<formal> formal_vector;

// Holds the parameters of a function and if they have default values
// (only applicable in some cases).
struct signature : public gc {
  formal_vector formals;

  // Formal for the rest parameter.  If there is no rest parameter, then the
  // type is null.
  formal rest;

  signature()
    : rest(0)
    {}

  virtual ~signature() {}

  void add(formal f) {
    formals.push_back(f);
  }

  void addRest(formal f) {
    rest=f;
  }

  bool hasRest() {
    return rest.t;
  }
  size_t getNumFormals() {
    return rest.t ? formals.size() + 1 : formals.size();
  }

  formal& getFormal(size_t n) {
    assert(n < formals.size());
    return formals[n];
  }

  formal& getRest() {
    return rest;
  }

  friend ostream& operator<< (ostream& out, const signature& s);

  friend bool equivalent(signature *s1, signature *s2);
#if 0
  friend bool castable(signature *target, signature *source);
  friend int numFormalsMatch(signature *s1, signature *s2);
#endif

  size_t hash();
};

struct function : public ty {
  ty *result;
  signature sig;

  function(ty *result)
    : ty(ty_function), result(result) {}
  function(ty *result, formal f1)
    : ty(ty_function), result(result) {
    add(f1);
  }
  function(ty *result, formal f1, formal f2)
    : ty(ty_function), result(result) {
    add(f1);
    add(f2);
  }
  virtual ~function() {}

  void add(formal f) {
    sig.add(f);
  }

  void addRest(formal f) {
    sig.addRest(f);
  }

  virtual bool isReference() {
    return true;
  }

  bool equiv(ty *other)
  {
    if (other->kind==ty_function) {
      function *that=(function *)other;
      return equivalent(this->result,that->result) &&
             equivalent(&this->sig,&that->sig);
    }
    else return false;
  }

  size_t hash() {
    return sig.hash()*0x1231+result->hash();
  }

  void print(ostream& out) const
    { out << *result << sig; }

  void printVar (ostream& out, symbol *name) const {
    result->printVar(out,name);
    out << sig;
  }

  ty *getResult() {
    return result;
  }
  
  signature *getSignature() {
    return &sig;
  }

#if 0
  ty *stripDefaults();
#endif

  // Initialized to null.
  trans::access *initializer();
};

typedef mem::vector<ty *> ty_vector;

// This is used in getType expressions when an overloaded variable is accessed.
class overloaded : public ty {
public:
  ty_vector sub;
public:
  overloaded()
    : ty(ty_overloaded) {}
  virtual ~overloaded() {}

  bool equiv(ty *other)
  {
    for(ty_vector::iterator i=sub.begin();i!=sub.end();++i)
      if (equivalent(*i,other))
        return true;
    return false;
  }

  size_t hash() {
    // Overloaded types should not be hashed.
    assert(False);
    return 0;
  }

  void add(ty *t) {
    if (t->kind == ty_overloaded) {
      overloaded *ot = (overloaded *)t;
      copy(ot->sub.begin(), ot->sub.end(),
	   inserter(this->sub, this->sub.end()));
    }
    else
      sub.push_back(t);
  }

  // Only add a type distinct from the ones currently in the overloaded type.
  // If special is false, just the distinct signatures are added.
  void addDistinct(ty *t, bool special=false);

  // If there are less than two overloaded types, the type isn't really
  // overloaded.  This gives a more appropriate type in this case.
  ty *simplify() {
    switch (sub.size()) {
      case 0:
	return 0;
      case 1: {
	return sub.front();
      }
      default:
	return new overloaded(*this);
    }
  }

  // Returns the signature-less type of the set.
  ty *signatureless();

  // True if one of the subtypes is castable.
  bool castable(ty *target, caster &c);

#if 0
  // This determines which type of function to use, given a signature of
  // types.  It is valid to have overloaded parameters here.
  // If exactly one type matches, it returns that type.
  // If no types match, it returns null.
  // Otherwise, it returns a new overloaded with all matches.
  ty *resolve(signature *key);
  ty *resolve(signature *key, symbol *name, position pos);
#endif

  // Use default printing for now.
};

// This is used to encapsulate iteration over the subtypes of an overloaded
// type.  The base method need only be implemented to handle non-overloaded
// types.
class collector {
public:
  virtual ~collector() {}
  virtual ty *base(ty *target, ty *source) = 0;

  virtual ty *collect(ty *target, ty *source) {
    if (overloaded *o=dynamic_cast<overloaded *>(target)) {
      ty_vector &sub=o->sub;

      overloaded *oo=new overloaded;
      for(ty_vector::iterator x = sub.begin(); x != sub.end(); ++x) {
        types::ty *t=collect(*x, source);
        if (t)
          oo->add(t);
      }

      return oo->simplify();
    }
    else if (overloaded *o=dynamic_cast<overloaded *>(source)) {
      ty_vector &sub=o->sub;

      overloaded *oo=new overloaded;
      for(ty_vector::iterator y = sub.begin(); y != sub.end(); ++y) {
        // NOTE: A possible speed optimization would be to replace this with a
        // call to base(), but this is only correct if we can guarantee that an
        // overloaded type has no overloaded sub-types.
        types::ty *t=collect(target, *y);
        if (t)
          oo->add(t);
      }

      return oo->simplify();
    }
    else
      return base(target, source);
  }
};

class tester {
public:
  virtual ~tester() {}
  virtual bool base(ty *target, ty *source) = 0;

  virtual bool test(ty *target, ty *source) {
    if (overloaded *o=dynamic_cast<overloaded *>(target)) {
      ty_vector &sub=o->sub;

      for(ty_vector::iterator x = sub.begin(); x != sub.end(); ++x)
        if (test(*x, source))
          return true;

      return false;
    }
    else if (overloaded *o=dynamic_cast<overloaded *>(source)) {
      ty_vector &sub=o->sub;

      for(ty_vector::iterator y = sub.begin(); y != sub.end(); ++y)
        if (base(target, *y))
          return true;

      return false;
    }
    else
      return base(target, source);
  }
};


} // namespace types

#endif
