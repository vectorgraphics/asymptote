/*****
 * types.cc
 * Andy Hammerlindl 2002/06/24
 *
 * Used by the compiler as a way to keep track of the type of a variable
 * or expression.
 *****/

#include <cstdio>
#include <algorithm>

#include "entry.h"
#include "types.h"
#include "runtime.h"
#include "access.h"

namespace types {

/* Base types */
#define PRIMITIVE(name,Name,asyName) \
  primitiveTy p##Name(ty_##name); \
  ty *prim##Name() { return &p##Name; }
#define PRIMERROR
#include <primitives.h>
#undef PRIMERROR
#undef PRIMITIVE
                             
nullTy pNull;
ty *primNull() { return &pNull; }

array boolArray_(primBoolean());
ty *boolArray() { return &boolArray_; }
array intArray_(primInt());
ty *intArray() { return &intArray_; }
array realArray_(primReal());
ty *realArray() { return &realArray_; }
array pairArray_(primPair());
ty *pairArray() { return &pairArray_; }
array tripleArray_(primTriple());
ty *tripleArray() { return &tripleArray_; }
array stringArray_(primString());
ty *stringArray() { return &stringArray_; }
array transformArray_(primTransform());
ty *transformArray() { return &transformArray_; }
array pathArray_(primPath());
ty *pathArray() { return &pathArray_; }
array penArray_(primPen());
ty *penArray() { return &penArray_; }
array guideArray_(primGuide());
ty *guideArray() { return &guideArray_; }
  
array boolArray2_(boolArray());
ty *boolArray2() { return &boolArray2_; }
array intArray2_(intArray());
ty *intArray2() { return &intArray2_; }
array realArray2_(realArray());
ty *realArray2() { return &realArray2_; }
array pairArray2_(pairArray());
ty *pairArray2() { return &pairArray2_; }
array tripleArray2_(tripleArray());
ty *tripleArray2() { return &tripleArray2_; }
array stringArray2_(stringArray());
ty *stringArray2() { return &stringArray2_; }
array penArray2_(penArray());
ty *penArray2() { return &penArray2_; }
  
array boolArray3_(boolArray2());
ty *boolArray3() { return &boolArray3_; }
array intArray3_(intArray2());
ty *intArray3() { return &intArray3_; }
array realArray3_(realArray2());
ty *realArray3() { return &realArray3_; }
array pairArray3_(pairArray2());
ty *pairArray3() { return &pairArray3_; }
array tripleArray3_(tripleArray2());
ty *tripleArray3() { return &tripleArray3_; }
array stringArray3_(stringArray2());
ty *stringArray3() { return &stringArray3_; }
  
const char *names[] = {
  "null",
  "<structure>", "<function>", "<overloaded>",
  
#define PRIMITIVE(name,Name,asyName) #asyName,
#define PRIMERROR
#include <primitives.h>
#undef PRIMERROR
#undef PRIMITIVE

  "<array>"
};

ty::~ty()
{}

void ty::print(ostream& out) const
{
  out << names[kind];
}

trans::varEntry *primitiveTy::virtualField(symbol *id, signature *sig)
{
#define FIELD(Type, name, func) \
  if (sig == 0 && id == symbol::trans(name)) { \
    static trans::bltinAccess a(run::func); \
    static trans::varEntry v(prim##Type(), &a, 0, position()); \
    return &v; \
  }

  switch (kind) {
    case ty_pair:
      FIELD(Real,"x",pairXPart);
      FIELD(Real,"y",pairYPart);
      break;
    case ty_triple:
      FIELD(Real,"x",tripleXPart);
      FIELD(Real,"y",tripleYPart);
      FIELD(Real,"z",tripleZPart);
      break;
    case ty_transform:
      FIELD(Real,"x",transformXPart);
      FIELD(Real,"y",transformYPart);
      FIELD(Real,"xx",transformXXPart);
      FIELD(Real,"xy",transformXYPart);
      FIELD(Real,"yx",transformYXPart);
      FIELD(Real,"yy",transformYYPart);
      break;
    case ty_tensionSpecifier:
      FIELD(Real,"out",tensionSpecifierOutPart);
      FIELD(Real,"in",tensionSpecifierInPart);
      FIELD(Boolean,"atLeast",tensionSpecifierAtleastPart);
      break;
    case ty_curlSpecifier:
      FIELD(Real,"value",curlSpecifierValuePart);
      FIELD(Int,"side",curlSpecifierSidePart);
      break;
    default:
      break;
  }
  return 0;

#undef FIELD
}

ty *ty::virtualFieldGetType(symbol *id)
{
  trans::varEntry *v = virtualField(id, 0);
  return v ? v->getType() : 0;
}

trans::access *nullTy::castTo(ty *target, caster &) {
  switch (target->kind) {
    case ty_array: {
      static trans::bltinAccess a(run::pushNullArray);
      return &a;
    }
    case ty_record: {
      static trans::bltinAccess a(run::pushNullRecord);
      return &a;
    } 
    case ty_function: {
      static trans::bltinAccess a(run::pushNullFunction);
      return &a;
    }
    default:
      return 0;
  }
}

trans::access *array::initializer()
{
  static trans::bltinAccess a(run::emptyArray);
  return &a;
}

ty *array::pushType()
{
  if (pushtype == 0)
    pushtype = new function(celltype,formal(celltype,"x"));

  return pushtype;
}

ty *array::popType()
{
  if (poptype == 0)
    poptype = new function(celltype);

  return poptype;
}

ty *array::appendType()
{
  if (appendtype == 0)
    appendtype = new function(primVoid(),formal(this,"a"));

  return appendtype;
}

ty *array::insertType()
{
  if (inserttype == 0) {
    function *f=new function(primVoid(),formal(primInt(),"i"));
    f->addRest(this);
    inserttype = f;
  }
  
  return inserttype;
}

ty *array::deleteType()
{
  if (deletetype == 0)
    deletetype = new function(primVoid(),formal(primInt(),"i",true),
			      formal(primInt(),"j",true));

  return deletetype;
}

ty *cyclicType() {
  return new function(primVoid(),formal(primBoolean(),"b"));
}

ty *initializedType() {
  return new function(primBoolean(),formal(primInt(),"i"));
}

ty *array::virtualFieldGetType(symbol *id)
{
  return
    id == symbol::trans("cyclic") ? cyclicType() : 
    id == symbol::trans("push") ? pushType() : 
    id == symbol::trans("pop") ? popType() : 
    id == symbol::trans("append") ? appendType() : 
    id == symbol::trans("insert") ? insertType() : 
    id == symbol::trans("delete") ? deleteType() : 
    id == symbol::trans("initialized") ? initializedType() : 
    ty::virtualFieldGetType(id);
}

trans::varEntry *array::virtualField(symbol *id, signature *sig)
{
  if (sig == 0 && id == symbol::trans("length"))
  {
    static trans::bltinAccess a(run::arrayLength);
    static trans::varEntry v(primInt(), &a, 0, position());
    return &v;
  }
  if (sig == 0 && id == symbol::trans("cyclicflag"))
  {
    static trans::bltinAccess a(run::arrayCyclicFlag);
    static trans::varEntry v(primBoolean(), &a, 0, position());
    return &v;
  }
  if (id == symbol::trans("cyclic") &&
      equivalent(sig, cyclicType()->getSignature()))
  {
    static trans::bltinAccess a(run::arrayCyclic);
    static trans::varEntry v(cyclicType(), &a, 0, position());
    return &v;
  }
  if (id == symbol::trans("initialized") &&
      equivalent(sig, initializedType()->getSignature()))
  {
    static trans::bltinAccess a(run::arrayInitialized);
    static trans::varEntry v(initializedType(), &a, 0, position());
    return &v;
  }
  if (id == symbol::trans("push") &&
      equivalent(sig, pushType()->getSignature()))
  {
    static trans::bltinAccess a(run::arrayPush);
    // v needs to be dynamic, as the push type differs among arrays.
    trans::varEntry *v = new trans::varEntry(pushType(), &a, 0, position());

    return v;
  }
  if (id == symbol::trans("pop") &&
      equivalent(sig, popType()->getSignature()))
  {
    static trans::bltinAccess a(run::arrayPop);
    // v needs to be dynamic, as the pop type differs among arrays.
    trans::varEntry *v = new trans::varEntry(popType(), &a, 0, position());

    return v;
  }
  if (id == symbol::trans("append") &&
      equivalent(sig, appendType()->getSignature()))
  {
    static trans::bltinAccess a(run::arrayAppend);
    // v needs to be dynamic, as the append type differs among arrays.
    trans::varEntry *v = new trans::varEntry(appendType(), &a, 0, position());

    return v;
  }
  if (id == symbol::trans("insert") &&
      equivalent(sig, insertType()->getSignature()))
  {
    static trans::bltinAccess a(run::arrayInsert);
    // v needs to be dynamic, as the insert type differs among arrays.
    trans::varEntry *v = new trans::varEntry(insertType(), &a, 0, position());

    return v;
  }
  if (id == symbol::trans("delete") &&
      equivalent(sig, deleteType()->getSignature()))
  {
    static trans::bltinAccess a(run::arrayDelete);
    // v needs to be dynamic, as the delete type differs among arrays.
    trans::varEntry *v = new trans::varEntry(deleteType(), &a, 0, position());

    return v;
  }
  else
    return ty::virtualField(id, sig);
}

std::ostream& operator<< (std::ostream& out, const formal& f)
{
  if (f.Explicit)
    out << "explicit ";
  if (f.name)
    f.t->printVar(out,f.name);
  else
    f.t->print(out);
  if (f.defval)
    out << "=<default>";
  return out;
}
  
bool equivalent(formal& f1, formal& f2) {
  // Just test the types.
  // This will also return true for the rest parameter if both types are null.
  // NOTE: Is this the right behavior?
  return equivalent(f1.t,f2.t);
}


std::ostream& operator<< (std::ostream& out, const signature& s)
{
  out << "(";

  formal_vector::const_iterator f = s.formals.begin();
  if (f != s.formals.end()) {
    out << *f;
    ++f;
  }
  for (; f != s.formals.end(); ++f)
    out << ", " << *f;

  if (s.rest.t) {
    if (!s.formals.empty())
      out << " ";
    out << "... " << s.rest;
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

  if (s1->formals.size() != s2->formals.size())
    return false;

  return std::equal(s1->formals.begin(),s1->formals.end(),s2->formals.begin(),
                    (bool (*)(formal&,formal&)) equivalent) &&
         equivalent(s1->rest, s2->rest);
}

size_t signature::hash() {
  size_t x=2038;
  for (formal_vector::iterator i=formals.begin(); i!=formals.end(); ++i)
    x=x*0xFAEC+i->t->hash();

  if (rest.t)
    x=x*0xACED +rest.t->hash();

  return x;
}

trans::access *function::initializer() {
  static trans::bltinAccess a(run::pushNullFunction);
  return &a;
}

#if 0
ty *function::stripDefaults()
{
  function *f = new function(result);

  int n = sig.getNumFormals();
  for (int i = 0; i < n; ++i)
    f->add(sig.getFormal(i), 0);

  return f;
}
#endif

// Only add a type with a signature distinct from the ones currently
// in the overloaded type.
void overloaded::addDistinct(ty *t, bool special)
{
  if (t->kind == ty_overloaded) {
    overloaded *ot = (overloaded *)t;
    for (ty_vector::iterator st = ot->sub.begin();
	 st != ot->sub.end();
	 ++st)
    {
      this->addDistinct(*st, special);
    }
  }
  else {
    for (ty_vector::iterator st = this->sub.begin();
	 st != this->sub.end();
	 ++st)
    {
      if (equivalent(t, *st, special))
	return;
    }

    // Nonequivalent in signature - add it.
    this->add(t);
  }
}


ty *overloaded::signatureless()
{
  for(ty_vector::iterator t = sub.begin(); t != sub.end(); ++t)
    if ((*t)->getSignature()==0)
      return *t;
 
  return 0;
}

bool overloaded::castable(ty *target, caster &c)
{
  for(ty_vector::iterator s = sub.begin(); s != sub.end(); ++s)
    if (c.castable(target,*s))
      return true;
  return false;
}

bool equivalent(ty *t1, ty *t2)
{
  // The same pointer must point to the same type.
  if (t1 == t2)
    return true; 

  // Handle empty types (used in equating empty rest parameters).
  if (t1 == 0 || t2 == 0)
    return false;

  // Ensure if an overloaded type is compared to a non-overloaded one, that the
  // overloaded type's method is called.
  if (t1->kind == ty_overloaded || t2->kind != ty_overloaded)
    return t1->equiv(t2);
  return t2->equiv(t1);
}

bool equivalent(ty *t1, ty *t2, bool special) {
  return special ? equivalent(t1, t2) :
                   equivalent(t1->getSignature(), t2->getSignature());
}

} // namespace types
