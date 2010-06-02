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
#include "runarray.h"
#include "runfile.h"
#include "runpair.h"
#include "runtriple.h"
#include "access.h"
#include "virtualfieldaccess.h"

namespace types {

/* Base types */
#define PRIMITIVE(name,Name,asyName)            \
  primitiveTy p##Name(ty_##name);               \
  ty *prim##Name() { return &p##Name; }         \
  array name##Array_(prim##Name());             \
  ty *name##Array() { return &name##Array_; }   \
  array name##Array2_(name##Array());           \
  ty *name##Array2() { return &name##Array2_; } \
  array name##Array3_(name##Array2());          \
  ty *name##Array3() { return &name##Array3_; }
#define PRIMERROR
#include "primitives.h"
#undef PRIMERROR
#undef PRIMITIVE
                             
nullTy pNull;
ty *primNull() { return &pNull; }
  
const char *names[] = {
  "null",
  "<structure>", "<function>", "<overloaded>",
  
#define PRIMITIVE(name,Name,asyName) #asyName,
#define PRIMERROR
#include "primitives.h"
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

// Used for primitive virtual fields and array virtual fields.
#define FIELD(Type, name, func)                                 \
  if (sig == 0 && id == symbol::trans(name)) {                  \
    static trans::virtualFieldAccess a(run::func);              \
    static trans::varEntry v(Type(), &a, 0, position());        \
    return &v;                                                  \
  }

#define RWFIELD(Type, name, getter, setter)                       \
  if (sig == 0 && id == symbol::trans(name)) {                    \
    static trans::virtualFieldAccess a(run::getter, run::setter); \
    static trans::varEntry v(Type(), &a, 0, position());          \
    return &v;                                                    \
  }
      
#define SIGFIELD(Type, name, func)                                      \
  if (id == symbol::trans(name) &&                                      \
      equivalent(sig, Type()->getSignature()))                          \
    {                                                                   \
      static trans::virtualFieldAccess a(run::func);                    \
      static trans::varEntry v(Type(), &a, 0, position());              \
      return &v;                                                        \
    }

#define DSIGFIELD(name, func)                                           \
  if (id == symbol::trans(#name) &&                                     \
      equivalent(sig, name##Type()->getSignature()))                    \
    {                                                                   \
      static trans::virtualFieldAccess a(run::func);                    \
      /* for some fields, v needs to be dynamic */                      \
      /* e.g. when the function type depends on an array type. */       \
      trans::varEntry *v =                                              \
        new trans::varEntry(name##Type(), &a, 0, position());           \
      return v;                                                         \
    }

#define FILEFIELD(GetType, SetType, name) \
  FIELD(GetType,#name,name##Part);        \
  SIGFIELD(SetType,#name,name##Set);


// TODO: Move to symbol.h
#define SYM( x ) symbol::trans(#x)

ty *dimensionType() {
  return new function(primFile(),
                      formal(primInt(),SYM(nx),true),
                      formal(primInt(),SYM(ny),true),
                      formal(primInt(),SYM(nz),true));
}

ty *modeType() {
  return new function(primFile(),formal(primBoolean(),SYM(b), true));
}

ty *readType() {
  return new function(primFile(), formal(primInt(), SYM(i)));
}

trans::varEntry *primitiveTy::virtualField(symbol *id, signature *sig)
{
  switch (kind) {
    case ty_pair:
      FIELD(primReal,"x",pairXPart);
      FIELD(primReal,"y",pairYPart);
      break;
    case ty_triple:
      FIELD(primReal,"x",tripleXPart);
      FIELD(primReal,"y",tripleYPart);
      FIELD(primReal,"z",tripleZPart);
      break;
    case ty_transform:
      FIELD(primReal,"x",transformXPart);
      FIELD(primReal,"y",transformYPart);
      FIELD(primReal,"xx",transformXXPart);
      FIELD(primReal,"xy",transformXYPart);
      FIELD(primReal,"yx",transformYXPart);
      FIELD(primReal,"yy",transformYYPart);
      break;
    case ty_tensionSpecifier:
      FIELD(primReal,"out",tensionSpecifierOutPart);
      FIELD(primReal,"in",tensionSpecifierInPart);
      FIELD(primBoolean,"atLeast",tensionSpecifierAtleastPart);
      break;
    case ty_curlSpecifier:
      FIELD(primReal,"value",curlSpecifierValuePart);
      FIELD(primInt,"side",curlSpecifierSidePart);
      break;
    case ty_file:      
      FIELD(primString,"name",namePart);
      FIELD(primString,"mode",modePart);
      FILEFIELD(IntArray,dimensionType,dimension);
      FILEFIELD(primBoolean,modeType,line);
      FILEFIELD(primBoolean,modeType,csv);
      FILEFIELD(primBoolean,modeType,word);
      FILEFIELD(primBoolean,modeType,singlereal);
      FILEFIELD(primBoolean,modeType,singleint);
      FILEFIELD(primBoolean,modeType,signedint);
      SIGFIELD(readType,"read",readSet);
      break;
    default:
      break;
  }
  return 0;
}

ty *overloadedDimensionType() {
  overloaded *o=new overloaded;
  o->add(dimensionType());
  o->add(IntArray());
  return o;
}

ty *overloadedModeType() {
  overloaded *o=new overloaded;
  o->add(modeType());
  o->add(primBoolean());
  return o;
}

ty *ty::virtualFieldGetType(symbol *id)
{
  trans::varEntry *v = virtualField(id, 0);
  return v ? v->getType() : 0;
}

ty *primitiveTy::virtualFieldGetType(symbol *id)
{
  if(kind == ty_file) {
    if (id == symbol::trans("dimension"))
      return overloadedDimensionType();
  
    if (id == symbol::trans("line") || id == symbol::trans("csv") || 
        id == symbol::trans("word") || id == symbol::trans("singlereal") || 
        id == symbol::trans("singleint") || id == symbol::trans("signedint"))
      return overloadedModeType();
  
    if (id == symbol::trans("read"))
      return readType();
  }
  
  trans::varEntry *v = virtualField(id, 0);
  
  return v ? v->getType() : 0;
}

#define RETURN_STATIC_BLTIN(func) \
  { \
    static trans::bltinAccess a(run::func); \
    return &a; \
  }

trans::access *nullTy::castTo(ty *target, caster &) {
  switch (target->kind) {
    case ty_array: {
       RETURN_STATIC_BLTIN(pushNullArray);
    }
    case ty_record: {
       RETURN_STATIC_BLTIN(pushNullRecord);
    } 
    case ty_function: {
       RETURN_STATIC_BLTIN(pushNullFunction);
    }
    default:
      return 0;
  }
}

trans::access *array::initializer()
{
  RETURN_STATIC_BLTIN(emptyArray)
}

ty *array::pushType()
{
  if (pushtype == 0)
    pushtype = new function(celltype,formal(celltype,SYM(x)));

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
    appendtype = new function(primVoid(),formal(this,SYM(a)));

  return appendtype;
}

ty *array::insertType()
{
  if (inserttype == 0) {
    function *f=new function(primVoid(),formal(primInt(),SYM(i)));
    f->addRest(this);
    inserttype = f;
  }
  
  return inserttype;
}

ty *array::deleteType()
{
  if (deletetype == 0)
    deletetype = new function(primVoid(),formal(primInt(),SYM(i),true),
                              formal(primInt(),SYM(j),true));

  return deletetype;
}

ty *initializedType() {
  return new function(primBoolean(),formal(primInt(),SYM(i)));
}

#define SIGFIELDLIST \
  ASIGFIELD(initialized, arrayInitialized); \
  ASIGFIELD(push, arrayPush); \
  ASIGFIELD(pop, arrayPop); \
  ASIGFIELD(append, arrayAppend); \
  ASIGFIELD(insert, arrayInsert); \
  ASIGFIELD(delete, arrayDelete); \

ty *array::virtualFieldGetType(symbol *id)
{
  #define ASIGFIELD(name, func) \
  if (id == symbol::trans(#name)) \
    return name##Type();

  SIGFIELDLIST

  #undef ASIGFIELD

  return ty::virtualFieldGetType(id);
}

trans::varEntry *array::virtualField(symbol *id, signature *sig)
{
  FIELD(primInt, "length", arrayLength);
  FIELD(IntArray, "keys", arrayKeys);
  RWFIELD(primBoolean, "cyclic", arrayCyclicFlag, arraySetCyclicFlag);

  #define ASIGFIELD(name, func) DSIGFIELD(name,func)
  
  SIGFIELDLIST
    
  #undef ASIGFIELD
  
  // Fall back on base class to handle no match.
  return ty::virtualField(id, sig);
}

#undef SIGFIELDLIST

ostream& operator<< (ostream& out, const formal& f)
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

bool argumentEquivalent(formal &f1, formal& f2) {
  if (f1.name == f2.name) {
    if (f1.t == 0)
      return f2.t == 0;
    else if (f2.t == 0)
      return false;

    return f1.t->kind != ty_overloaded &&
      f2.t->kind != ty_overloaded &&
      equivalent(f1.t, f2.t);
  }
  else
    return false;
}

ostream& operator<< (ostream& out, const signature& s)
{
  if (s.isOpen) {
    out << "(<open>)";
    return out;
  }

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

  // Two open signatures are always equivalent, as the formals are ignored.
  if (s1->isOpen)
    return s2->isOpen;
  else if (s2->isOpen)
    return false;

  if (s1->formals.size() != s2->formals.size())
    return false;

  return std::equal(s1->formals.begin(),s1->formals.end(),s2->formals.begin(),
                    (bool (*)(formal&,formal&)) equivalent) &&
    equivalent(s1->rest, s2->rest);
}

bool argumentEquivalent(signature *s1, signature *s2)
{
  // Handle null signature
  if (s1 == 0)
    return s2 == 0;
  else if (s2 == 0)
    return false;

  if (s1->formals.size() != s2->formals.size())
    return false;

  return std::equal(s1->formals.begin(),s1->formals.end(),s2->formals.begin(),
                    (bool (*)(formal&,formal&)) argumentEquivalent) &&
    argumentEquivalent(s1->rest, s2->rest);
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
  RETURN_STATIC_BLTIN(pushNullFunction);
}

#if 0
ty *function::stripDefaults()
{
  function *f = new function(result);

  Int n = sig.getNumFormals();
  for (Int i = 0; i < n; ++i)
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

#undef FIELD
#undef RWFIELD
#undef SIGFIELD
#undef DSIGFIELD

} // namespace types
