/*****
 * types.h
 * Andy Hammerlindl 2002/06/20
 *
 * Used by the compiler as a way to keep track of the type of a variable
 * or expression.
 *
 * NOTE: This isn't very object oriented, might be improvable to be more
 * robust.
 *****/

#ifndef TYPES_H
#define TYPES_H

#include <iostream>
#include <cstdio>
#include <cassert>
#include <vector>

#include "pool.h"
#include "symbol.h"

using std::ostream;
using std::vector;

using sym::symbol;

// Forward declaration.
namespace trans {
class varEntry;
}
namespace absyntax {
class varinit;
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
  ty_transform,
  ty_guide,
  ty_path,
  ty_pen,
  ty_picture,	// "frame" in Asymptote language
  ty_file,
  ty_array
};

// Forward declarations.
struct ty;
struct signature;

// Checks if two types are equal in the sense of the language.
// That is primitive types are equal if they are the same kind.
// Structures are equal if they come from the same struct definition.
// Arrays are equal if their cell types are equal.
bool equivalent(ty *t1, ty *t2);

class ty : public memory::managed<ty> {
public:
  const ty_kind kind;
  ty(ty_kind kind)
    : kind(kind) {}
  virtual ~ty();

  virtual void print (ostream& out) const;

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

  // For pair's x and y, and array's length, this is a special type of
  // "field".
  // In actually, it returns a function which takes the object as its
  // parameter and returns the necessary result.
  // These should not have public permission, as modifying them would
  // have strange results.
  virtual trans::varEntry *virtualField(symbol *, signature *);

  // varGetType for virtual fields.
  // Unless you are using functions for virtual fields, this should work
  // fine.
  virtual ty *virtualFieldGetType(symbol *id);

  // Returns the type.  In case of functions, return the equivalent type
  // but with no default values for parameters.
  virtual ty *stripDefaults()
  {
    return this;
  }

  // Returns true if the other type is equivalent to this one.
  // The general function equivalent should be preferably used, as it properly
  // handles overloaded type comparisons.
  virtual bool equiv(ty *other)
  {
    return this==other;
  }
};

class primitiveTy : public ty 
{
public:
  primitiveTy(ty_kind kind)
    : ty(kind) {}
  
  virtual bool primitive() {
    return true;
  }

  virtual bool isReference() {
    return kind==ty_null;
  }
  
  bool equiv(ty *other)
  {
    return this->kind==other->kind;
  }
};

// Ostream output, just defer to print.
inline ostream& operator<< (ostream& out, const ty& t)
  { t.print(out); return out; }

struct array : public ty {
  ty *celltype;
  ty *pushtype;

  array(ty *celltype)
    : ty(ty_array), celltype(celltype), pushtype(0) {}

  virtual bool isReference() {
    return true;
  }

  bool equiv(ty *other) {
    return other->kind==ty_array &&
           equivalent(this->celltype,((array *)other)->celltype);
  }

  void print(ostream& out) const
    { out << *celltype << "[]"; }

  ty *pushType();

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
ty *primTransform();
ty *primGuide();
ty *primPath();
ty *primPen();
ty *primPicture();
ty *primFile();
  
ty *boolArray();
ty *intArray();
ty *realArray();
ty *pairArray();
ty *stringArray();
ty *pathArray();
ty *penArray();
  
ty *guideArray();

ty *boolArray2();
ty *intArray2();
ty *realArray2();
ty *pairArray2();
ty *stringArray2();

ty *boolArray3();
ty *intArray3();
ty *realArray3();
ty *pairArray3();
ty *stringArray3();
  
// Holds the parameters of a function and if they have default values
// (only applicable in some cases).  Technically, a signature should
// also hold the function name.
class signature : public memory::managed<signature> {
  vector<ty *> formals;

  // Holds the index of the expression in an array of default
  // expressions.
  vector<absyntax::varinit*> defaults;
  size_t ndefault;

  vector<bool> Explicit;
public:
  signature()
    : ndefault(0) {} 

  virtual ~signature()
    {}

  void add(ty *t, absyntax::varinit *def=0, bool Explicit=false)
    { 
      formals.push_back(t);
      this->Explicit.push_back(Explicit);
      defaults.push_back(def);
      if(def) ++ndefault;
    }

  int getNumFormals()
    { return (int) formals.size(); }

  ty *getFormal(int n) {
    assert((unsigned)n < formals.size());
    return formals[n];
  }

  absyntax::varinit *getDefault(size_t n) {
    assert(n < defaults.size());
    return defaults[n];
  }

  bool getExplicit(size_t n) {
    return Explicit[n];
  }

  friend ostream& operator<< (ostream& out, const signature& s);

  friend bool equivalent(signature *s1, signature *s2);
  friend bool castable(signature *target, signature *source);
  friend int numFormalsMatch(signature *s1, signature *s2);
};

struct function : public ty {
  ty *result;
  signature sig;

  function(ty *result)
    : ty(ty_function), result(result) {}
  virtual ~function() {}

  void add(ty *t, absyntax::varinit *def=0, bool Explicit=false) {
    sig.add(t, def, Explicit);
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

  void print(ostream& out) const
    { out << *result << ' ' << sig; }

  ty *getResult() {
    return result;
  }
  
  signature *getSignature() {
    return &sig;
  }

  ty *stripDefaults();
};

// This is used in getType expressions when it is an overloaded
// varible being  accessed.
class overloaded : public ty {
public:
  typedef std::vector<ty *> ty_vector;
  typedef ty_vector::iterator ty_iter;
  ty_vector sub;
public:
  overloaded()
    : ty(ty_overloaded) {}
  virtual ~overloaded() {}

  bool equiv(ty *other)
  {
    for(ty_iter i=sub.begin();i!=sub.end();++i)
      if (equivalent(*i,other))
        return true;
    return false;
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

  // Only add a type with a signature distinct from the ones currently
  // in the overloaded type.
  void addDistinct(ty *t);

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

  // This determines which type of function to use, given a signature of
  // types.  It is valid to have overloaded parameters here.
  // If exactly one type matches, it returns that type.
  // If no types match, it returns null.
  // Otherwise, it returns a new overloaded with all matches.
  ty *resolve(signature *key);

  // Use default printing for now.
};

} // namespace types

#endif
