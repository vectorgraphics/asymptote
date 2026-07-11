/*****
 * builtin_handlers.cc
 *
 * Custom call-site handlers for open-signature builtins.  Each such builtin
 * is registered (via addOpenBuiltinFunc in builtin.cc) with a pair of member
 * function pointers on callExp -- one for trans, one for getType -- that
 * override the usual dispatch when overload resolution picks the open
 * signature.  This file centralizes those handlers along with the
 * symbol-keyed side table that maps a builtin's symbol to its handlers.
 *****/

#include "exp.h"
#include "errormsg.h"
#include "runtime.h"
#include "runarray.h"
#include "runmath.h"
#include "coenv.h"
#include "application.h"
#include "inst.h"
#include "opsymbols.h"
#include "access.h"
#include "callable.h"
#include "stack.h"

namespace absyntax {

using namespace types;
using namespace trans;
using vm::inst;

namespace {

bltin aliasCompareBuiltin(ty *target)
{
  switch (target->kind) {
    case ty_array:
      return run::arrayAlias;
    case ty_function:
      return run::boolFuncEq;
    default:
      return run::boolMemEq;
  }
}

bltin aliasNullBuiltin(ty *target)
{
  switch (target->kind) {
    case ty_array:
      return run::arrayAliasNull;
    case ty_function:
      return run::boolFuncEqNull;
    default:
      return run::boolMemEqNull;
  }
}

// If `target` is the function signature `bool(T, T)` with a single common
// operand type T and no defaults / rest / open formals, return T; otherwise
// return nullptr.  Shared by the record `==`/`!=` and `alias` specialize hooks.
types::ty *binaryBoolOperandType(ty *target)
{
  function *ft = dynamic_cast<function *>(target);
  if (!ft) return nullptr;
  if (!equivalent(ft->getResult(), primBoolean())) return nullptr;
  signature *sig = ft->getSignature();
  if (!sig || sig->isOpen || sig->hasRest()) return nullptr;
  if (sig->getNumFormals() != 2) return nullptr;
  types::ty *t0 = sig->getFormal(0).t;
  types::ty *t1 = sig->getFormal(1).t;
  if (!t0 || !t1 || !equivalent(t0, t1)) return nullptr;
  return t0;
}

// Tests whether `target` has the function signature `bool(R, R)` with R a
// concrete record type.  Returns the record type on success, nullptr otherwise.
record *recordEqTarget(ty *target)
{
  types::ty *t = binaryBoolOperandType(target);
  if (!t || t->kind != ty_record) return nullptr;
  return dynamic_cast<record *>(t);
}

} // namespace

trans::access *specializeRecordEq(ty *target)
{
  if (!recordEqTarget(target)) return nullptr;
  return new trans::callableAccess(new vm::bfunc(run::boolMemEq));
}

trans::access *specializeRecordNeq(ty *target)
{
  if (!recordEqTarget(target)) return nullptr;
  return new trans::callableAccess(new vm::bfunc(run::boolMemNeq));
}

trans::access *specializeAlias(ty *target)
{
  // `alias` may be coerced to `bool(T, T)` for any nullable (reference) type T
  // --- records, arrays, and function types --- with the runtime comparison
  // selected the same way as at a direct call site.
  types::ty *t = binaryBoolOperandType(target);
  if (!t || !t->isReference() || t->kind == ty_null) return nullptr;
  return new trans::callableAccess(new vm::bfunc(aliasCompareBuiltin(t)));
}

namespace {

// Symbol-keyed side table of custom call-site handlers.  See the comment on
// registerCustomHandlers() in exp.h.
struct symbolHash {
  size_t operator()(symbol s) const { return s.hash(); }
};
struct symbolEq {
  bool operator()(symbol s, symbol t) const { return s == t; }
};
using HandlerMap =
  mem::unordered_map<symbol, callExp::CustomHandlers, symbolHash, symbolEq>;

HandlerMap &customHandlersTable()
{
  static HandlerMap table;
  return table;
}

// Find a unique record type that occurs in both `lt` and `rt` (each of which
// may be a concrete type or an overloaded set).  Returns null on no match or
// ambiguous match.
record *commonRecordType(ty *lt, ty *rt)
{
  overloaded ltFlat, rtFlat, common;
  ltFlat.add(lt);
  rtFlat.add(rt);
  for (ty *lty : ltFlat.sub) {
    if (lty->kind != ty_record) continue;
    for (ty *rty : rtFlat.sub) {
      if (rty->kind != ty_record) continue;
      if (equivalent(lty, rty))
        common.addDistinct(lty);
    }
  }
  ty *resolved = common.simplify();
  if (!resolved || resolved->kind != ty_record) return nullptr;
  return dynamic_cast<record *>(resolved);
}

// Find the unique record type referenced by `t` (which may be overloaded).
// Returns null if none, or more than one, record type is present.
record *uniqueRecordType(ty *t)
{
  overloaded flat;
  flat.add(t);
  record *found = nullptr;
  for (ty *sub : flat.sub) {
    if (sub->kind != ty_record) continue;
    record *r = dynamic_cast<record *>(sub);
    if (!r) continue;
    if (found && !equivalent(found, r)) return nullptr;
    found = r;
  }
  return found;
}

// Validate that the operand expressions of a `==`/`!=` call resolve to a
// common record type, with null allowed on at most one side.  Returns the
// target record type on success, or null on failure (without emitting a
// diagnostic, so the caller can decide whether and how to report the error).
record *resolveRecordEqTarget(exp *left, exp *right, coenv &e)
{
  types::ty *lt = left->getType(e);
  types::ty *rt = right->getType(e);

  if (lt->kind == ty_error || rt->kind == ty_error) return nullptr;

  if (lt->kind == ty_null && rt->kind == ty_null) return nullptr;
  if (lt->kind == ty_null) return uniqueRecordType(rt);
  if (rt->kind == ty_null) return uniqueRecordType(lt);
  return commonRecordType(lt, rt);
}

} // namespace

void registerCustomHandlers(symbol name, callExp::CustomHandlers h)
{
  // Idempotent: base_venv() may run more than once over the lifetime of the
  // process (e.g. once for the per-process genv and once for runtime
  // initialization).  Subsequent calls must register the same handlers.
  HandlerMap &table = customHandlersTable();
  auto it = table.find(name);
  if (it != table.end()) {
    assert(it->second.trans == h.trans && it->second.getType == h.getType &&
           it->second.specialize == h.specialize);
    return;
  }
  table[name] = h;
}

const callExp::CustomHandlers *lookupCustomHandlers(symbol name)
{
  if (!name) return nullptr;
  HandlerMap &table = customHandlersTable();
  auto it = table.find(name);
  return it == table.end() ? nullptr : &it->second;
}

types::ty *callExp::getAliasType(coenv &)
{
  // Invoked from callExp::getType only after open-signature resolution has
  // already been confirmed by the dispatch wrapper, so we simply return the
  // alias result type.
  return primBoolean();
}

types::ty *callExp::transAlias(coenv &e)
{
  // Called only when the alias builtin (OPEN signature) has been resolved.
  // Apply strict same-type, nullable semantics.
  cachedApp = 0;
  cachedVarEntry = 0;

  if (args->rest.val || args->size() != 2) {
    em.error(getPos());
    em << "alias takes exactly two parameters";
    return primError();
  }

  exp *left = (*args)[0].val;
  exp *right = (*args)[1].val;
  types::ty *lt = left->getType(e);
  types::ty *rt = right->getType(e);

  if (lt->kind == ty_error || rt->kind == ty_error) {
    reportArgErrors(e);
    return primError();
  }

  if (lt->kind == ty_null && rt->kind == ty_null) {
    em.error(getPos());
    em << "alias is not defined when both arguments are null literals";
    return primError();
  }

  ty *target;
  if (lt->kind == ty_null) {
    target = rt;
    if (target->isOverloaded()) {
      em.error(getPos());
      em << "alias: argument type is ambiguous";
      return primError();
    }
  } else if (rt->kind == ty_null) {
    target = lt;
    if (target->isOverloaded()) {
      em.error(getPos());
      em << "alias: argument type is ambiguous";
      return primError();
    }
  } else {
    // Both non-null: find the intersection of concrete types from lt and rt.
    // This handles overloaded variable names (same name, multiple types in
    // scope) --- e.g., alias(c, f) where c is overloaded{B, real(real)} and
    // f is real(real) resolves unambiguously to real(real).
    overloaded ltFlat, rtFlat, common;
    ltFlat.add(lt);  // flattens overloaded; no-op for concrete types
    rtFlat.add(rt);
    for (ty *lty : ltFlat.sub)
      for (ty *rty : rtFlat.sub)
        if (equivalent(lty, rty))
          common.addDistinct(lty);

    ty *resolved = common.simplify();
    if (!resolved) {
      em.error(getPos());
      em << "alias requires both arguments to have the same type";
      return primError();
    }
    if (resolved->kind == ty_overloaded) {
      em.error(getPos());
      em << "alias: argument type is ambiguous";
      return primError();
    }
    target = resolved;
  }

  if (!target->isReference() || target->kind == ty_null) {
    em.error(getPos());
    em << "alias requires arguments of a nullable type";
    return primError();
  }

  // Emit the comparison.  Use transToType to handle overloaded variable
  // types (where the same name is in scope multiple times); the type
  // equivalence check above already guarantees no actual cast occurs.
  if (lt->kind == ty_null) {
    right->transToType(e, target);
    e.c.encode(inst::builtin, aliasNullBuiltin(target));
  } else if (rt->kind == ty_null) {
    left->transToType(e, target);
    e.c.encode(inst::builtin, aliasNullBuiltin(target));
  } else {
    left->transToType(e, target);
    right->transToType(e, target);
    e.c.encode(inst::builtin, aliasCompareBuiltin(target));
  }

  return primBoolean();
}

types::ty *callExp::getRecordEqType(coenv &e)
{
  // Return primBoolean() only when the operands can actually be reduced to a
  // record comparison; otherwise return ty_error so that equalityExp::trans
  // can fall through to its function-equality special case (and so that any
  // other "no matching == for these types" diagnostic propagates correctly).
  if (args->rest.val || args->size() != 2) return primError();
  return resolveRecordEqTarget((*args)[0].val, (*args)[1].val, e)
         ? primBoolean() : primError();
}

types::ty *callExp::transRecordEq(coenv &e)
{
  // Invoked when `==` or `!=` resolves to the open-signature builtin (i.e.
  // when no concrete equality is in scope for the operand types).  Apply
  // record reference-equality semantics, allowing null on at most one side.
  cachedApp = 0;
  cachedVarEntry = 0;

  symbol op = callee->getName();
  bool isEq = (op == SYM_EQ);
  assert(isEq || op == SYM_NEQ);

  if (args->rest.val || args->size() != 2) {
    em.error(getPos());
    em << "'" << op << "' takes exactly two operands";
    return primError();
  }

  exp *left = (*args)[0].val;
  exp *right = (*args)[1].val;

  // Surface argument errors first.
  types::ty *lt = left->getType(e);
  types::ty *rt = right->getType(e);
  if (lt->kind == ty_error || rt->kind == ty_error) {
    reportArgErrors(e);
    return primError();
  }

  record *target = resolveRecordEqTarget(left, right, e);
  if (!target) {
    em.error(getPos());
    em << "no matching '" << op << "' for these operand types";
    return primError();
  }

  // Emit the comparison.
  left->transToType(e, target);
  right->transToType(e, target);
  e.c.encode(inst::builtin, isEq ? run::boolMemEq : run::boolMemNeq);

  return primBoolean();
}

} // namespace absyntax
