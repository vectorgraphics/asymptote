/*****
 * cast.cc
 * Andy Hammerlindl 2003/07/24
 *
 * Handles the casting of types, numeric promotions, and the operators,
 * using a table of accesses for each.  An access here specifies the
 * instruction or builtin function used to handle the casting.
 *****/

#include <map>

#include "cast.h"
#include "runtime.h"
#include "access.h"
#include "builtin.h"

using namespace trans;
using namespace vm;

namespace types {

using trans::access;
using camp::pair;
using camp::path;
  
typedef unsigned int uint;

const uint firstPrim = (uint)ty_boolean;
const uint lastPrim = (uint)ty_array;

const uint numKinds = lastPrim + 1;

// The tables that keep track of the accesses
access *inits[numKinds];
  
const uint arrayDepth=4; // Accept casting of arrays of depth < arrayDepth
  
// Casts:  
access *ecasts[arrayDepth][arrayDepth][numKinds][numKinds];
access *casts[arrayDepth][arrayDepth][numKinds][numKinds];
access *promotions[arrayDepth][arrayDepth][numKinds][numKinds];
  
// Operators are stored via a map.
// If this proves too inefficient, a different scheme may be used.
struct opKey {
  int ltypecode;
  int rtypecode;
  int symcode;
};

// The identity access, ie. no instructions are encoded for a cast, and
// no fuctions are called.
identAccess id;


static bool valid(ty *target)
{
  uint n = (uint)target->kind;
  return (firstPrim <= n && n <= lastPrim);
}

int getindex(uint& i, uint& n1, ty *target)
{
  i=0;
  while((n1=target->kind) == ty_array) {
    target=((array *) target)->celltype;
    i++;
    if(i >= arrayDepth) return -1;
  }
  return 0;
}
  
void addCast(access *A[arrayDepth][arrayDepth][numKinds][numKinds],
	     ty *target, ty*source, access *a)
{
  uint i,j,n1,n2;
  if(getindex(i,n1,target)) return;
  if(getindex(j,n2,source)) return;
  A[i][j][n1][n2]=a;
}
  
access *Cast(access *A[arrayDepth][arrayDepth][numKinds][numKinds],
	     ty *target, ty*source)
{
  uint i,j,n1,n2;
  if(getindex(i,n1,target)) return 0;
  if(getindex(j,n2,source)) return 0;
  return A[i][j][n1][n2];
}
  
// Specifies that source may be cast to target, but only if an explicit
// cast expression is used.
void addExplicitCast(ty *target, ty *source, access *a)
{
  assert(valid(target) && valid(source));
  addCast(ecasts,target,source,a);
}

// Specifies that source may be implicitly cast to target by the
// function or instruction described in a.
// Also add it to explicit cast.
void addCast(ty *target, ty *source, access *a)
{
  assert(valid(target) && valid(source));
  addExplicitCast(target,source,a);
  addCast(casts,target,source,a);
}

// Specifies that source may be promoted to target in binary
// expressions.
// Also adds it to cast and explicit cast.
void addPromotion(ty *target, ty *source, access *a, bool castToo=true)
{
  assert(valid(target) && valid(source));
  if(castToo) addCast(target,source,a);
  addCast(promotions,target,source,a);
}

void addExplicitCast(ty *target, ty *source, bltin f)
{
  access *a = new bltinAccess(f);
  addExplicitCast(target, source, a);
}

void addCast(ty *target, ty *source, bltin f)
{
  access *a = new bltinAccess(f);
  addCast(target, source, a);
}

void addPromotion(ty *target, ty *source, bltin f, bool castToo=true)
{
  access *a = new bltinAccess(f);
  addPromotion(target, source, a, castToo);
}

void initializeCasts()
{
  addExplicitCast(primInt(), primReal(), run::cast<double,int>);
  addExplicitCast(primString(), primInt(), run::stringCast<int>);
  addExplicitCast(primString(), primReal(), run::stringCast<double>);
  addExplicitCast(primString(), primPair(), run::stringCast<pair>);
  addExplicitCast(primInt(), primString(), run::castString<int>);
  addExplicitCast(primReal(), primString(), run::castString<double>);
  addExplicitCast(primPair(), primString(), run::castString<pair>);

  addPromotion(primReal(), primInt(), run::cast<int,double>);
  addPromotion(primPair(), primInt(), run::cast<int,pair>);
  addPromotion(primPair(), primReal(), run::cast<double,pair>);
  
  addPromotion(primPath(), primPair(), run::cast<pair,path>);
  addPromotion(primGuide(), primPair(), run::pairToGuide);
  addPromotion(primGuide(), primPath(), run::pathToGuide);
  addCast(primPath(), primGuide(), run::guideToPath);

  addCast(primBoolean(), primFile(), run::read<bool>);
  addCast(primInt(), primFile(), run::read<int>);
  addCast(primReal(), primFile(), run::read<double>);
  addCast(primPair(), primFile(), run::read<pair>);
  addCast(primString(), primFile(), run::read<string>);
  
  addExplicitCast(intArray(), realArray(), run::arrayToArray<double,int>);
  
  addPromotion(realArray(), intArray(), run::arrayToArray<int,double>);
  addPromotion(pairArray(), intArray(), run::arrayToArray<int,pair>);
  addPromotion(pairArray(), realArray(), run::arrayToArray<double,pair>);
  
  addCast(boolArray(), primFile(), run::readArray<bool>);
  addCast(intArray(), primFile(), run::readArray<int>);
  addCast(realArray(), primFile(), run::readArray<double>);
  addCast(pairArray(), primFile(), run::readArray<pair>);
  addCast(stringArray(), primFile(), run::readArray<string>);
  
  addCast(boolArray2(), primFile(), run::readArray<bool>);
  addCast(intArray2(), primFile(), run::readArray<int>);
  addCast(realArray2(), primFile(), run::readArray<double>);
  addCast(pairArray2(), primFile(), run::readArray<pair>);
  addCast(stringArray2(), primFile(), run::readArray<string>);
  
  addCast(boolArray3(), primFile(), run::readArray<bool>);
  addCast(intArray3(), primFile(), run::readArray<int>);
  addCast(realArray3(), primFile(), run::readArray<double>);
  addCast(pairArray3(), primFile(), run::readArray<pair>);
  addCast(stringArray3(), primFile(), run::readArray<string>);
}

void addInitializer(ty *t, access *a)
{
  assert(valid(t));
  inits[t->kind] = a;
}

void addInitializer(ty *t, bltin f)
{
  access *a = new bltinAccess(f);
  addInitializer(t, a);
}

void initializeInitializers()
{
  addInitializer(primBoolean(), run::boolFalse);
  addInitializer(primInt(), run::intZero);
  addInitializer(primReal(), run::realZero);

  addInitializer(primString(), run::emptyString);
  addInitializer(primPair(), run::pairZero);
  addInitializer(primTransform(), run::transformIdentity);
  addInitializer(primGuide(), run::nullGuide);
  addInitializer(primPath(), run::nullPath);
  addInitializer(primPen(), run::defaultpen);
  addInitializer(primPicture(), run::nullFrame);
  addInitializer(primFile(), run::nullFile);
}

// Gets the initializer for a type.
// NOTE: There may be a better place for this than with the casts.
access *initializer(ty *t)
{
  if (t->primitive())
    return inits[t->kind];
  else if (t->kind == ty_array) {
    static bltinAccess a(run::emptyArray);
    return &a;
  } else if (t->kind == ty_record) {
    // NOTE: May want to allocate a new instance of the record instead.
    static bltinAccess a(run::pushNullRecord);
    return &a;
  } else if (t->kind == ty_function) {
    static bltinAccess a(run::pushNullFunction);
    return &a;
  } else
    return 0;
}

bool castable(ty *target, ty *source)
{
  // If errors already exist, don't report more.
  // This may, however, cause problems with resoving the signature of an
  // overloaded function to use.  The abstract syntax should check if any
  // of the parameters had an error before finding the signature.
  if (target->kind == ty_error || source->kind == ty_error)
    return true;
 
  // Identity conversion
  if (target == source)
    return true;

  // Casting to an overloaded type should never be considered.
  assert(target->kind != ty_overloaded);

  // Casting of overloaded types is based on a match of subtypes.
  if (source->kind == ty_overloaded) {
    overloaded *set = (overloaded *)source;
    for (vector<ty *>::iterator t = set->sub.begin();
         t != set->sub.end();
	 ++t) {
     if (castable(target, *t))
       return true;
    }

    // No matches.
    return false;
  }
  
  // Hopefully, nobody is trying to do this.
  if (target->kind == ty_void || target->kind == ty_null)
    return false;

  // Check array equivalence.
  if (target->kind == ty_array)
  {
    if (source->kind == ty_array) {
      if(equivalent(((array *)target)->celltype,
		    ((array *)source)->celltype)) return true;
    }
    else if (source->kind == ty_null)
      return true;
  }

  // Null to reference cast
  if (target->kind == ty_record) {
    if (source->kind == ty_null)
      return true;
    else
      return false;
  }

  // Function casting (for now) allows only equivalent function types.
  // NOTE: consider constructing casts for castable functions.
  if (target->kind == ty_function) {
    if (source->kind == ty_function) {
      function *f1 = (function *)target, *f2 = (function *)source;
      return equivalent(f1->result, f2->result) &&
             equivalent(&f1->sig, &f2->sig);
    }
    else if (source->kind == ty_null)
      return true;
    else
      return false;
  }

  // Primative kinds of the same type
  //if (target->kind == source->kind)
  //  return true;
  
  // Otherwise primitive casts must be looked up in the table
  
  return Cast(casts,target,source) ? true : false;
}


ty *castType(ty *target, ty *source)
{
  // If errors already exist, don't report more.
  // This may, however, cause problems with resoving the signature of an
  // overloaded function to use.  The abstract syntax should check if any
  // of the parameters had an error before finding the signature.
  if (target->kind == ty_error || source->kind == ty_error)
    return primError();
 
  // Identity conversion
  if (target == source)
    return target;

  // Casting of overloaded types is based on a match of subtypes.
  if (overloaded *set = dynamic_cast<overloaded *>(target)) {
    overloaded result;

    for (vector<ty *>::iterator t = set->sub.begin();
         t != set->sub.end();
	 ++t) {
      if (castable(*t, source))
        result.add(*t);
    }
    return result.simplify();
  }
  if (source->kind == ty_overloaded) {
    overloaded *set = dynamic_cast<overloaded *>(source);

    for (vector<ty *>::iterator t = set->sub.begin();
         t != set->sub.end();
	 ++t) {
     if (castable(target, *t))
       return target;
    }

    // No matches.
    return 0;
  }
  
  // Hopefully, nobody is trying to do this.
  if (target->kind == ty_void || target->kind == ty_null)
    return 0;

  // Check array equivalence.
  if (target->kind == ty_array)
  {
    if (source->kind == ty_array) {
      if(equivalent(((array *)target)->celltype,
		    ((array *)source)->celltype)) return target;
    }
    else if (source->kind == ty_null)
      return target;
  }

  // Null to reference cast
  if (target->kind == ty_record) {
    if (source->kind == ty_null)
      return target;
    else
      return 0;
  }

  // Function casting (for now) allows only equivalent function types.
  // NOTE: consider constructing casts for castable functions.
  if (target->kind == ty_function) {
    if (source->kind == ty_function) {
      function *f1 = (function *)target, *f2 = (function *)source;
      return equivalent(f1->result, f2->result) &&
             equivalent(&f1->sig, &f2->sig)
	     ? target : 0;
    }
    else if (source->kind == ty_null)
      return target;
    else
      return 0;
  }

  // Primative kinds of the same type
  //if (target->kind == source->kind)
  //  return true;
  
  // Otherwise primitive casts must be looked up in the table
					 
  return Cast(casts,target,source) ? target : 0;
}


ty *explicitCastType(ty *target, ty *source)
{
  // If errors already exist, don't report more.
  // This may, however, cause problems with resoving the signature of an
  // overloaded function to use.  The abstract syntax should check if any
  // of the parameters had an error before finding the signature.
  if (target->kind == ty_error || source->kind == ty_error)
    return primError();
 
  // Identity conversion
  if (target == source)
    return target;

  // Casting of overloaded types is based on a match of subtypes.
  if (target->kind == ty_overloaded) {
    overloaded *set = (overloaded *)target;
    overloaded result;

    for (vector<ty *>::iterator t = set->sub.begin();
         t != set->sub.end();
	 ++t) {
      types::ty *ts = explicitCastType(*t, source);
      if (ts)
        result.add(*t);
    }
    return result.simplify();
  }
  if (source->kind == ty_overloaded) {
    overloaded *set = (overloaded *)source;
    overloaded result;

    for (vector<ty *>::iterator t = set->sub.begin();
         t != set->sub.end();
	 ++t) {
      types::ty *ts = explicitCastType(target, *t);
      if (ts)
        result.add(ts);
    }

    return result.simplify();
  }
  
  // Hopefully, nobody is trying to do this.
  if (target->kind == ty_void || target->kind == ty_null)
    return 0;

  // Check array equivalence.
  if (target->kind == ty_array)
  {
    if (source->kind == ty_array) {
      if(equivalent(((array *)target)->celltype,
		    ((array *)source)->celltype)) return target;
    }
    else if (source->kind == ty_null)
      return target;
  }

  // Null to reference cast
  if (target->kind == ty_record) {
    if (source->kind == ty_null)
      return target;
    else
      return 0;
  }

  // Function casting (for now) allows only equivalent function types.
  // NOTE: consider constructing casts for castable functions.
  if (target->kind == ty_function) {
    if (source->kind == ty_function) {
      function *f1 = (function *)target, *f2 = (function *)source;
      return equivalent(f1->result, f2->result) &&
             equivalent(&f1->sig, &f2->sig)
	     ? target : 0;
    }
    else if (source->kind == ty_null)
      return target;
    else
      return 0;
  }

  // Primative kinds of the same type
  //if (target->kind == source->kind)
  //  return true;
  
  // Otherwise primitive casts must be looked up in the table
  // Notice that the source type is returned as that is the translated
  // type before the explicit cast.
  
  return Cast(ecasts,target,source) ? source : 0;
}


access *explicitCast(ty *target, ty *source)
{
  // If errors already exist, don't report more.
  // This may, however, cause problems with resolving the signature of an
  // overloaded function to use.  The abstract syntax should check if any
  // of the parameters had an error before finding the signature.
  if (target->kind == ty_error || source->kind == ty_error)
    return &id;
 
  // Identity conversion
  if (target == source)
    return &id;

  // Overloaded types should only be used in resolving function and
  // operation, not in actual translation.
  assert(target->kind != ty_overloaded &&
         source->kind != ty_overloaded);

  // Hopefully, nobody is trying to do this.
  if (target->kind == ty_void || target->kind == ty_null)
    return 0;

  // Check array equivalence.
  if (target->kind == ty_array)
  {
    if (source->kind == ty_array) {
      if(equivalent(((array *)target)->celltype,
		    ((array *)source)->celltype)) return &id;
    }
    else if (source->kind == ty_null) {
      static bltinAccess a(run::pushNullArray);
      return &a;
    }
  }

  // Null to reference cast
  if (target->kind == ty_record) {
    if (source->kind == ty_null) {
      static bltinAccess a(run::pushNullRecord);
      return &a;
    } else
      return 0;
  }

  // Function casting (for now) allows only equivalent function types.
  // NOTE: consider constructing casts for castable functions.
  if (target->kind == ty_function) {
    if (source->kind == ty_function) {
      function *f1 = (function *)target, *f2 = (function *)source;
      return equivalent(f1->result, f2->result) &&
             equivalent(&f1->sig, &f2->sig)
	     ? &id : 0;
    }
    else if (source->kind == ty_null) {
      static bltinAccess a(run::pushNullRecord);
      return &a;
    } else
      return 0;
  }

  // Primative kinds of the same type
  //if (target->kind == source->kind)
  //  return &id;
  
  // Otherwise primitive casts must be looked up in the table
      
  return Cast(ecasts,target,source);
}

access *cast(ty *target, ty *source)
{
  // If errors already exist, don't report more.
  // This may, however, cause problems with resoving the signature of an
  // overloaded function to use.  The abstract syntax should check if any
  // of the parameters had an error before finding the signature.
  if (target->kind == ty_error || source->kind == ty_error)
    return &id;
 
  // Identity conversion
  if (target == source)
    return &id;

  // Overloaded types should only be used in resolving function and
  // operation, not in actual translation.
  assert(target->kind != ty_overloaded &&
         source->kind != ty_overloaded);

  // Hopefully, nobody is trying to do this.
  if (target->kind == ty_void || target->kind == ty_null)
    return 0;

  // Check array equivalence.
  if (target->kind == ty_array)
  {
    if (source->kind == ty_array) {
      if(equivalent(((array *)target)->celltype,
		    ((array *)source)->celltype)) return &id;
    }
    else if (source->kind == ty_null) {
      static bltinAccess a(run::pushNullArray);
      return &a;
    }
  }

  // Null to reference cast
  if (target->kind == ty_record) {
    if (source->kind == ty_null) {
      static bltinAccess a(run::pushNullRecord);
      return &a;
    } else
      return 0;
  }

  // Function casting (for now) allows only equivalent function types.
  // NOTE: consider constructing casts for castable functions.
  if (target->kind == ty_function) {
    if (source->kind == ty_function) {
      function *f1 = (function *)target, *f2 = (function *)source;
      return equivalent(f1->result, f2->result) &&
             equivalent(&f1->sig, &f2->sig)
	     ? &id : 0;
    }
    else if (source->kind == ty_null) {
      static bltinAccess a(run::pushNullFunction);
      return &a;
    } else
      return 0;
  }

  // Primative kinds of the same type
  //if (target->kind == source->kind)
  //  return &id;
  
  // Otherwise primitive casts must be looked up in the table

  return Cast(casts,target,source);
}

ty *promote(ty *t1, ty *t2)
{
  if (t1->kind == ty_error || t2->kind == ty_error)
    return primError();
 
  // Identity
  if (t1 == t2)
    return t1;

  // Overloaded types have overloaded promotions.
  if (t1->kind == ty_overloaded) {
    overloaded *set = dynamic_cast<overloaded *>(t1);
    overloaded result;

    for (vector<ty *>::iterator t = set->sub.begin();
         t != set->sub.end();
	 ++t) {
      ty *pt = promote(*t, t2);
      if (pt)
	result.add(pt);
    }

    return result.simplify();
  }
  if (t2->kind == ty_overloaded) {
    overloaded *set = (overloaded *)t2;
    overloaded result;
    
    for (vector<ty *>::iterator t = set->sub.begin();
         t != set->sub.end();
	 ++t) {
      ty *pt = promote(t1, *t);
      if (pt)
	result.add(pt);
    }

    return result.simplify();
  }

  // Hopefully, nobody is trying to do this.
  if (t1->kind == ty_void || t2->kind == ty_void)
    return 0;

  // Check null to reference conversion
  if (t1->kind == ty_null) {
    if (t2->kind == ty_null)
      return t1;
    else if (t2->kind == ty_record ||
             t2->kind == ty_array ||
	     t2->kind == ty_function)
      return t2;
    else
      return 0;
  }
  if (t2->kind == ty_null) {
    if (t1->kind == ty_record ||
        t1->kind == ty_array ||
	t1->kind == ty_function)
      return t1;
    else
      return 0;
  }

  // Check array equivalence.
  if (t1->kind == ty_array)
  {
    if (t2->kind == ty_array) {
      if(equivalent(((array *)t1)->celltype,
		    ((array *)t2)->celltype)) return t1;
    }
  }
  
  // Function casting (for now) allows only equivalent function types.
  // NOTE: consider constructing casts for castable functions.
  if (t1->kind == ty_function) {
    if (t2->kind == ty_function) {
      function *f1 = (function *)t1, *f2 = (function *)t2;
      if (equivalent(f1->result, f2->result) &&
          equivalent(&f1->sig, &f2->sig))
        return f1->stripDefaults();
      else
        return 0;
    }
    else
      return 0;
  }

  // Now, only primitive conversions are allowed.
  if (!valid(t1) || !valid(t2))
    return 0;

  // Primative kinds of the same type
  //if (t1->kind == t2->kind)
  //  return t1;

  // Otherwise primitive promotions must be looked up in the table
  
  // If t2 can be promoted to t1, t1 is the final type.
  if(Cast(promotions,t1,t2))
    return t1;
  
  // ... and vice versa.
  if(Cast(promotions,t2,t1))
    return t2;
  
  // Otherwise, no promotion is possible.
  return 0;
}

} // namespace types
