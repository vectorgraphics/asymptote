/*****
 * types.cc
 * Andy Hammerlindl 2002/06/24
 *
 * Used by the compiler as a way to keep track of the type of a variable
 * or expression.
 *****/

#include <cstdio>

#include "entry.h"
#include "types.h"
#include "cast.h"
#include "runtime.h" // For arrayLength().
#include "access.h"

namespace types {

/* Base types */
ty pVoid(ty_void);
ty *primVoid() { return &pVoid; }
ty pNull(ty_null);
ty *primNull() { return &pNull; }
ty pError(ty_error);
ty *primError() { return &pError; }
primitiveTy pBoolean(ty_boolean);
ty *primBoolean() { return &pBoolean; }
primitiveTy pInt(ty_int);
ty *primInt() { return &pInt; }
primitiveTy pReal(ty_real);
ty *primReal() { return &pReal; }
primitiveTy pString(ty_string);
ty *primString() { return &pString; }
primitiveTy pPair(ty_pair);
ty *primPair() { return &pPair; }
primitiveTy pTransform(ty_transform);
ty *primTransform() { return &pTransform; }
primitiveTy pGuide(ty_guide);
ty *primGuide() { return &pGuide; }
primitiveTy pPath(ty_path);
ty *primPath() { return &pPath; }
primitiveTy pPen(ty_pen);
ty *primPen() { return &pPen; }
primitiveTy pPicture(ty_picture);
ty *primPicture() { return &pPicture; }
primitiveTy pFile(ty_file);
ty *primFile() { return &pFile; }
  
array boolArray_(primBoolean());
ty *boolArray() { return &boolArray_; }
array intArray_(primInt());
ty *intArray() { return &intArray_; }
array realArray_(primReal());
ty *realArray() { return &realArray_; }
array pairArray_(primPair());
ty *pairArray() { return &pairArray_; }
array stringArray_(primString());
ty *stringArray() { return &stringArray_; }
array pathArray_(primPath());
ty *pathArray() { return &pathArray_; }
  
array boolArray2_(boolArray());
ty *boolArray2() { return &boolArray2_; }
array intArray2_(intArray());
ty *intArray2() { return &intArray2_; }
array realArray2_(realArray());
ty *realArray2() { return &realArray2_; }
array pairArray2_(pairArray());
ty *pairArray2() { return &pairArray2_; }
array stringArray2_(stringArray());
ty *stringArray2() { return &stringArray2_; }
  
array boolArray3_(boolArray2());
ty *boolArray3() { return &boolArray3_; }
array intArray3_(intArray2());
ty *intArray3() { return &intArray3_; }
array realArray3_(realArray2());
ty *realArray3() { return &realArray3_; }
array pairArray3_(pairArray2());
ty *pairArray3() { return &pairArray3_; }
array stringArray3_(stringArray2());
ty *stringArray3() { return &stringArray3_; }
  
const char *names[] = {
  "void", "null",
  "<structure>", "<function>", "<error>", "<overloaded>",
  "bool", "int", "real",
  "string",
  "pair", "transform", "guide", "path", "pen", "frame",
  "file",
  "<array>"
};

ty::~ty()
{}

void ty::print(ostream& out) const
{
  out << names[kind];
}

trans::varEntry *ty::virtualField(symbol *id, signature *sig)
{
  if (primitive())
    switch (kind) {
      case ty_pair:
        if (sig == 0 && id == symbol::trans("x"))
        {
          static trans::bltinAccess a(run::pairXPart);
          static trans::varEntry v(primReal(), &a);

          return &v;
        }
        if (sig == 0 && id == symbol::trans("y"))
        {
          static trans::bltinAccess a(run::pairYPart);
          static trans::varEntry v(primReal(), &a);

          return &v;
        }
        //TODO: Add transform.
      default:
        return 0;
    }
  else
    return 0;
}

ty *ty::virtualFieldGetType(symbol *id)
{
  trans::varEntry *v = virtualField(id, 0);
  return v ? v->getType() : 0;
}

ty *array::pushType()
{
  if (pushtype == 0) {
    function *ft = new function(primVoid());
    ft->add(celltype);
    pushtype = ft;
  }

  return pushtype;
}

ty *array::virtualFieldGetType(symbol *id)
{
  return id == symbol::trans("push") ? pushType() : ty::virtualFieldGetType(id);
}

trans::varEntry *array::virtualField(symbol *id, signature *sig)
{
  if (sig == 0 && id == symbol::trans("length"))
  {
    static trans::bltinAccess a(run::arrayLength);
    static trans::varEntry v(primInt(), &a);
    return &v;
  }
  if (id == symbol::trans("push") &&
      equivalent(sig, pushType()->getSignature()))
  {
    static trans::bltinAccess a(run::arrayPush);
    // v needs to be dynamic, as the push type differs among arrays.
    trans::varEntry *v = new trans::varEntry(pushType(), &a);

    return v;
  }
  else
    return ty::virtualField(id, sig);
}


std::ostream& operator<< (std::ostream& out, const signature& s)
{
  out << "(";

  vector<ty *>::const_iterator t = s.formals.begin();
  vector<absyn::varinit*>::const_iterator dt = s.defaults.begin();
  if (t != s.formals.end()) {
    out << **t;
    if (*dt != 0)
      out << " = <default>"; 
    ++t; ++dt;
  }
  for (; t != s.formals.end(); ++t, ++dt) {
    out << ", " << **t;
    if (*dt != 0)
      out << " = <default>"; 
  }

  out << ")";

  return out;
}


// Equivalence by design does not look at the presence of default values.
bool equivalent(signature *s1, signature *s2)
{
  // Handle null signature
  if (s1 == 0)
    return s2 == 0;
  else if (s2 == 0)
    return false;

  unsigned int n = (unsigned int)s1->formals.size();
  if (n != s2->formals.size())
    return false;

  for (unsigned int i = 0; i < n; i++)
    if (!equivalent(s1->formals[i], s2->formals[i]))
      return false;

  return true;
}

bool castable(signature *target, signature *source)
{
  // Handle null signature
  if (target == 0)
    return source == 0;
  else if (source == 0)
    return false;

  unsigned int m = (unsigned int)target->formals.size();
  unsigned int n = (unsigned int)source->formals.size();
  if (n > m || n+target->ndefault < m) return false;

  unsigned int j=0;
  for (unsigned int i = 0; i < m; i++) {
    if (j < n && (target->Explicit[i] ? 
		  equivalent(target->formals[i], source->formals[j]) :
		  castable(target->formals[i], source->formals[j])))
      j++;
    else if (!target->defaults[i]) return false;
  }
  
  return (j == n) ? true : false;
}


int numFormalsMatch(signature *target, signature *source)
{
  unsigned int m = (unsigned int)target->formals.size();
  unsigned int n = (unsigned int)source->formals.size();

  int matches = 0;
  for (unsigned int i = 0, j = 0; i < m; i++) {
    if (j < n && castable(target->formals[i], source->formals[j])) {
      if (equivalent(target->formals[i], source->formals[j]))
        matches++;
      j++;
    }
  }
  
  return matches;
}

ty *function::stripDefaults()
{
  function *f = new function(result);

  int n = sig.getNumFormals();
  for (int i = 0; i < n; ++i)
    f->add(sig.getFormal(i), 0);

  return f;
}

// Only add a type with a signature distinct from the ones currently
// in the overloaded type.
void overloaded::addDistinct(ty *t)
{
  if (t->kind == ty_overloaded) {
    overloaded *ot = (overloaded *)t;
    for (vector<ty *>::iterator st = ot->sub.begin();
	 st != ot->sub.end();
	 ++st)
    {
      this->addDistinct(*st);
    }
  }
  else {
    signature *tsig = t->getSignature();
    for (vector<ty *>::iterator st = this->sub.begin();
	 st != this->sub.end();
	 ++st)
    {
      if (equivalent(tsig, (*st)->getSignature()))
	return;
    }

    // Nonequivalent in signature - add it.
    this->add(t);
  }
}


	 

ty *overloaded::resolve(signature *key)
{
  overloaded set;
  
  // Pick out all applicable signatures.
  for(vector<ty *>::iterator t = sub.begin();
      t != sub.end();
      ++t)
  {
    signature *nsig = (*t)->getSignature();
   
    if (castable(nsig, key)) {
      set.add(*t);

      // Shortcut for simple (ie. non-function) variables.
      if (key == 0)
	return (*t);
    }
  }

  vector<ty *>& candidates = set.sub;
  if (candidates.size() <= 1)
    return set.simplify();

  // Try to further resolve candidates by checking for number of
  // arguments exactly matched.
  for (int n = key->getNumFormals(); n > 0; n--)
  {
    vector<ty *> matches;
    for (vector<ty *>::iterator p = candidates.begin();
         p != candidates.end();
	 ++p)
    {
      if (numFormalsMatch((*p)->getSignature(), key) >= n) {
        matches.push_back(*p);
      }
    }

    if (matches.size() == 1)
      return matches.front();
    if (matches.size() > 1)
      break;
  }

  return new overloaded(set);
}

bool equivalent(ty *t1, ty *t2)
{
  // The same pointer must point to the same type.
  if (t1 == t2)
    return true; 

  // Ensure if an overloaded type is compared to a non-overloaded one, that the
  // overloaded type's method is called.
  if (t1->kind == ty_overloaded || t2->kind != ty_overloaded)
    return t1->equiv(t2);
  else
    return t2->equiv(t1);
}

} // namespace types
