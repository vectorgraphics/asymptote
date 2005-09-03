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
using trans::action;
using trans::READ;
using trans::WRITE;
using trans::CALL;
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


void name::forceEquivalency(action act, coenv &e,
                            types::ty *target, types::ty *source)
{
  if (act == READ)
    e.implicitCast(getPos(), target, source);
  else if (!equivalent(target, source)) {
    em->compiler(getPos());
    *em << "type mismatch in variable: "
      << *target
      << " vs " << *source;
  }
}

frame *name::frameTrans(coenv &e)
{
  types::ty *t=signatureless(varGetType(e));

  if (t && t->kind == types::ty_record) {
    varTrans(READ, e, t);
    return ((record *)t)->getLevel();
  }
  return 0;
}

types::ty *name::getType(coenv &e, bool tacit)
{
  types::ty *t=signatureless(varGetType(e));
  return t ? t : typeTrans(e, tacit);
}


void simpleName::varTrans(action act, coenv &e, types::ty *target)
{
  //varEntry *v = e.e.lookupExactVar(id, target->getSignature());
  varEntry *v = e.e.lookupVarByType(id, target);
  
  if (v) {
    if (checkAmbiguity(getPos(), id, v)) {
      // TODO: Check permissions.
      v->getLocation()->encode(act, getPos(), e.c);
      forceEquivalency(act, e, target, v->getType());
    }
  }
  else {
    em->error(getPos());
    *em << "no matching variable of name \'" << *id << "\'";
  }
}

types::ty *simpleName::varGetType(coenv &e)
{
  return e.e.varGetType(id);
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

bool qualifiedName::varTransVirtual(action act, coenv &e,
                                    types::ty *target, types::ty *qt)
{
  varEntry *v = qt->virtualField(id, target->getSignature());
  if (v) {
    // Push qualifier onto stack.
    qualifier->varTrans(READ, e, qt);

    if (act == WRITE) {
      em->error(getPos());
      *em << "virtual field '" << *id << "' of '" << *qt
          << "' cannot be modified";
    }
    else {
      // Call instead of reading as it is a virtual field.
      v->getLocation()->encode(CALL, getPos(), e.c);
      e.implicitCast(getPos(), target, v->getType());

      if (act == CALL)
        // In this case, the virtual field will construct a vm::func object
        // and push it on the stack.
        // Then, pop and call the function.
        e.c.encode(inst::popcall);
    }

    // A virtual field was used.
    return true;
  }

  // No virtual field.
  return false;
}

void qualifiedName::varTransField(action act, coenv &e,
                                  types::ty *target, record *r)
{
  //v = r->lookupExactVar(id, target->getSignature());
  varEntry *v = r->lookupVarByType(id, target);

  if (v) {
    // TODO: Add permission checking.
    access *loc = v->getLocation();

    frame *f = qualifier->frameTrans(e);
    if (f)
      loc->encode(act, getPos(), e.c, f);
    else
      loc->encode(act, getPos(), e.c);

    forceEquivalency(act, e, target, v->getType());
  }
  else {
    em->error(getPos());
    *em << "no matching field of name \'" << *id << "\' in \'" << *r << "\'";
  }
}

void qualifiedName::varTrans(action act, coenv &e, types::ty *target)
{
  types::ty *qt = qualifier->getType(e);

  // Use virtual fields if applicable.
  if (varTransVirtual(act, e, target, qt))
    return;

  record *r = getRecord(qt);
  if (r)
    varTransField(act, e, target, r);
}

types::ty *qualifiedName::varGetType(coenv &e)
{
  types::ty *qt = qualifier->getType(e, true);

  // Look for virtual fields.
  types::ty *t = qt->virtualFieldGetType(id);
  if (t)
    return t;

  record *r = getRecord(qt, true);
  return r ? r->varGetType(id) : 0;
}

types::ty *qualifiedName::typeTrans(coenv &e, bool tacit)
{
  types::ty *rt = qualifier->typeTrans(e, tacit);

  record *r = getRecord(rt, tacit);
  if (!r)
    return primError();

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

trans::import *qualifiedName::typeGetImport(coenv &e)
{
  return qualifier->typeGetImport(e);
}

void qualifiedName::prettyprint(ostream &out, int indent)
{
  prettyindent(out, indent);
  out << "qualifiedName '" << *id << "'\n";

  qualifier->prettyprint(out, indent+1);
}

} // namespace absyntax
