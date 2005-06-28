/*****
 * name.cc
 * Andy Hammerlindl2002/07/14
 *
 * Qualified names (such as x, f, builtin.sin, a.b.c.d, etc.) can be used
 * either as varibles or a type names.  This class stores qualified
 * names used in nameExp and nameTy in the abstract syntax, and
 * implements the exp and type functions.
 *****/

#include "name.h"
#include "frame.h"
#include "record.h"
#include "coenv.h"
#include "inst.h"

namespace absyntax {
using namespace types;
using trans::access;
using vm::inst;


// Checks if a varEntry returned from coenv::lookupExactVar is ambiguous,
// an reports an error if it is.
static bool checkAmbiguity(position pos, symbol *s, varEntry *v)
{
  types::ty *t = v->getType();
  assert(t);

  if (t->kind == types::ty_overloaded) {
    em->error(pos);
    *em << "variable of name \'" << *s << "\' is ambiguous";
    return false;
  }
  else
    // All is well
    return true;
}

types::ty *signatureless(types::ty *t) {
  if (overloaded *o=dynamic_cast<overloaded *>(t))
    return o->signatureless();
  else
    return (t && !t->getSignature()) ? t : 0;
}

types::ty *simpleName::getType(coenv &e, bool tacit)
{
  types::ty *t=signatureless(e.e.varGetType(id));

  if (t)
    return t;
  else {
    types::ty *t = e.e.lookupType(id);
    if (t) {
      if (t->kind == types::ty_overloaded) {
	if (!tacit) {
	  em->error(getPos());
	  *em << "type of name \'" << *id << "\' is ambiguous";
	}
	return primError();
      }
      return t;
    }
    else {
      // NOTE: Should call getModule or something here.
      if (!tacit) {
        em->error(getPos());
        *em << "no variable or type of name \'" << *id << "\'";
      }
      return primError();
    }
  }
}

void simpleName::varTrans(coenv &e, types::ty *target)
{
  //varEntry *v = e.e.lookupExactVar(id, target->getSignature());
  varEntry *v = e.e.lookupVarByType(id, target);
  
  if (v) {
    if (checkAmbiguity(getPos(), id, v)) {
      v->getLocation()->encodeRead(getPos(), e.c);
      e.implicitCast(getPos(), target, v->getType());
    }
  }
  else {
    em->error(getPos());
    *em << "no matching variable of name \'" << *id << "\'";
  }
}

void simpleName::varTransWrite(coenv &e, types::ty *target)
{
  //varEntry *v = e.e.lookupExactVar(id, target->getSignature());
  varEntry *v = e.e.lookupVarByType(id, target);

  if (v) {
    if (checkAmbiguity(getPos(), id, v)) {
      v->getLocation()->encodeWrite(getPos(), e.c);
      if (!equivalent(v->getType(), target)) {
	em->compiler(getPos());
	*em << "type mismatch in variable writing: "
	    << *(v->getType())
	    << " vs " << *target;
      }
    }
  }
  else {
    em->error(getPos());
    *em << "no matching variable of name \'" << *id << "\'";
  }
}

void simpleName::varTransCall(coenv &e, types::ty *target)
{
  //varEntry *v = e.e.lookupExactVar(id, target->getSignature());
  varEntry *v = e.e.lookupVarByType(id, target);

  if (v) {
    if (checkAmbiguity(getPos(), id, v)) {
      v->getLocation()->encodeCall(getPos(), e.c);
      if (!equivalent(v->getType(), target)) {
	em->compiler(getPos());
	*em << "type mismatch in variable call";
      }
    }
  }
  else {
    em->error(getPos());
    *em << "no matching variable of name \'" << *id << "\'";
  }
}

types::ty *simpleName::varGetType(coenv &e)
{
  types::ty *t = e.e.varGetType(id);
  return t ? t : primError();
}

types::ty *simpleName::typeTrans(coenv &e, bool tacit)
{
  types::ty *t = e.e.lookupType(id);
  if (t) {
    if (t->kind == types::ty_overloaded) {
      if (!tacit) {
	em->error(getPos());
	*em << "type of name \'" << *id << "\' is ambiguous";
      }
      return primError();
    }
    return t;
  }
  else {
    // NOTE: Could call getModule here.
    if (!tacit) {
      em->error(getPos());
      *em << "no type of name \'" << *id << "\'";
    }
    return primError();
  }
}

trans::import *simpleName::typeGetImport(coenv &e)
{
  return e.e.lookupTypeImport(id);
}

frame *simpleName::frameTrans(coenv &e)
{
  types::ty *t=signatureless(e.e.varGetType(id));
  //varEntry *v = e.e.lookupExactVar(id, 0);

  if (t && t->kind == types::ty_record) {
    varEntry *v = e.e.lookupVarByType(id, t);
    assert(v);
    v->getLocation()->encodeRead(getPos(), e.c);
    return ((record *)t)->getLevel();
  }
  return 0;
}

void simpleName::prettyprint(ostream &out, int indent)
{
  prettyindent(out, indent);
  out << "simpleName '" << *id << "'\n";
}


record *qualifiedName::getRecord(types::ty *t, bool tacit)
{
  switch (t->kind) {
    case ty_overloaded:
      if (!tacit) {
        em->compiler(qualifier->getPos());
        *em << "name::getType returned overloaded";
      }
      return 0;
    case ty_record:
      return (record *)t;
    case ty_error:
      return 0;
    default:
      if (!tacit) {
        em->error(qualifier->getPos());
        *em << "type \'" << *t << "\' is not a structure";
      }
      return 0;
  }
}

types::ty *qualifiedName::getType(coenv &e, bool tacit)
{
  types::ty *qt = qualifier->getType(e, tacit);

  // Look for virtual fields.
  types::ty *vt = qt->virtualFieldGetType(id);
  if (vt)
    return vt;
 
  // Convert to a record. 
  record *r = getRecord(qt, tacit);
  if (!r)
    return primError();

  //varEntry *v = r->lookupExactVar(id, 0);

  types::ty *t=signatureless(r->varGetType(id));
  
  if (t)
    return t;
  else {
    types::ty *t = r->lookupType(id);
    if (t) {
      if (t->kind == types::ty_overloaded) {
	if (!tacit) {
	  em->error(getPos());
	  *em << "type of name \'" << *id << "\' is ambiguous";
	}
	return primError();
      }
      return t;
    }
    else {
      if (!tacit) {
	em->error(getPos());
	*em << "no matching field or type of name \'" << *id << "\' in \'"
	    << *r << "\'";
      }
      return primError();
    }
  }
}

void qualifiedName::varTrans(coenv &e, types::ty *target)
{
  types::ty *qt = qualifier->getType(e);

  // Look for virtual fields.
  varEntry *v = qt->virtualField(id, target->getSignature());
  if (v) {
    // Push qualifier onto stack.
    qualifier->varTrans(e, qt);

    // Call instead of reading as it is a virtual field.
    v->getLocation()->encodeCall(getPos(), e.c);
    e.implicitCast(getPos(), target, v->getType());
    return;
  }

  record *r = getRecord(qt);
  if (!r)
    return;

  //v = r->lookupExactVar(id, target->getSignature());
  v = r->lookupVarByType(id, target);

  if (v) {
    access *loc = v->getLocation();

    frame *f = qualifier->frameTrans(e);
    if (f)
      loc->encodeRead(getPos(), e.c, f);
    else
      loc->encodeRead(getPos(), e.c);

    e.implicitCast(getPos(), target, v->getType());
  }
  else {
    em->error(getPos());
    *em << "no matching field of name \'" << *id << "\' in \'" << *r << "\'";
  }
}

void qualifiedName::varTransWrite(coenv &e, types::ty *target)
{
  types::ty *qt = qualifier->getType(e);

  // Look for virtual fields.
  varEntry *v = qt->virtualField(id, target->getSignature());
  if (v) {
    // Push qualifier onto stack.
    qualifier->varTrans(e, qt);
    
    em->error(getPos());
    *em << "virtual field '" << *id << "' of '" << *qt
        << "' cannot be modified";
  }
 
  record *r = getRecord(qt);
  if (!r)
    return;

  //v = r->lookupExactVar(id, target->getSignature());
  v = r->lookupVarByType(id, target);

  if (v) {
    access *loc = v->getLocation();

    frame *f = qualifier->frameTrans(e);
    if (f)
      loc->encodeWrite(getPos(), e.c, r->getLevel());
    else
      loc->encodeWrite(getPos(), e.c);

    if (!equivalent(v->getType(), target)) {
      em->compiler(getPos());
      *em << "type mismatch in variable writing";
    }
  }
  else {
    em->error(getPos());
    *em << "no matching field of name \'" << *id << "\' in \'"
	<< *r << "\'";
  }
}

void qualifiedName::varTransCall(coenv &e, types::ty *target)
{
  types::ty *qt = qualifier->getType(e);

  // Look for virtual fields.
  varEntry *v = qt->virtualField(id, target->getSignature());
  if (v) {
    // Push qualifier onto stack.
    qualifier->varTrans(e, qt);
    
    // Call instead of reading as it is a virtual field.
    v->getLocation()->encodeCall(getPos(), e.c);
    e.implicitCast(getPos(), target, v->getType());

    // In this case, the virtual field will construct a vm::func object
    // and push it on the stack.
    // Then, pop and call the function.
    e.c.encode(inst::popcall);
    return;
  }

  record *r = getRecord(qt);
  if (!r)
    return;

  //v = r->lookupExactVar(id, target->getSignature());
  v = r->lookupVarByType(id, target);

  if (v) {
    access *loc = v->getLocation();
    
    frame *f = qualifier->frameTrans(e);
    if (f)
      loc->encodeCall(getPos(), e.c, f);
    else
      loc->encodeCall(getPos(), e.c);

    if (!equivalent(v->getType(), target)) {
      em->compiler(getPos());
      *em << "type mismatch in variable call";
    }
  }
  else {
    em->error(getPos());
    *em << "no matching field of name \'" << *id << "\' in \'"
	<< *r << "\'";
  }
}

types::ty *qualifiedName::varGetType(coenv &e)
{
  types::ty *qt = qualifier->getType(e, true);

  // Look for virtual fields.
  types::ty *t = qt->virtualFieldGetType(id);
  if (t)
    return t;

  record *r = getRecord(qt, true);
  if (r) {
    types::ty *t = r->varGetType(id);
    return t ? t : primError();
  }
  else
    return primError();
}

types::ty *qualifiedName::typeTrans(coenv &e, bool tacit)
{
  types::ty *rt = qualifier->typeTrans(e, tacit);
  if (rt->kind == ty_error)
    return rt;
  else if (rt->kind != ty_record) {
    if (!tacit) {
      em->error(getPos());
      *em << "type '" << *rt << "' is not a structure";
    }
    return primError();
  }

  record *r = (record *)rt;

  types::ty *t = r->lookupType(id);
  if (t) {
    if (t->kind == types::ty_overloaded) {
      if (!tacit) {
	em->error(getPos());
	*em << "type of name \'" << *id << "\' is ambiguous";
      }
      return primError();
    }
    return t;
  }
  else {
    if (!tacit) {
      em->error(getPos());
      *em << "no type of name \'" << *id << "\' in \'"
          << *r << "\'";
    }
    return primError();
  }
}

trans::import *qualifiedName::typeGetImport(coenv &e)
{
  return qualifier->typeGetImport(e);
}

frame *qualifiedName::frameTrans(coenv &e)
{
  types::ty *qt = qualifier->getType(e, true);
  record *r = getRecord(qt, true);
  if (!r)
    return 0;

  //varEntry *v = r->lookupExactVar(id, 0);
  types::ty *t=signatureless(r->varGetType(id));

  if (t && t->kind == types::ty_record) {
    varEntry *v=r->lookupVarByType(id, t);
    access *a = v->getLocation();
    
    frame *level = qualifier->frameTrans(e);
    if (level)
      a->encodeRead(getPos(), e.c, level);
    else
      a->encodeRead(getPos(), e.c);

    return ((record *)t)->getLevel();
  }
  return 0;
}

void qualifiedName::prettyprint(ostream &out, int indent)
{
  prettyindent(out, indent);
  out << "qualifiedName '" << *id << "'\n";

  qualifier->prettyprint(out, indent+1);
}



} // namespace absyntax
