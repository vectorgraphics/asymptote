/*****
 * fundec.h
 * Andy Hammerlindl 2002/8/29
 *
 * Defines the semantics for defining functions.  Both the newexp syntax, and
 * the abbreviated C-style function definition.
 *****/

#include "fundec.h"
#include "errormsg.h"
#include "coenv.h"
#include "stm.h"
#include "runtime.h"

namespace absyntax {

using namespace trans;
using namespace types;

void formal::prettyprint(ostream &out, int indent)
{
  prettyname(out, "formal",indent);
  
  base->prettyprint(out, indent+1);
  if (start) start->prettyprint(out, indent+1);
  if (defval) defval->prettyprint(out, indent+1);
}

types::formal formal::trans(coenv &e, bool encodeDefVal, bool tacit) {
  return types::formal(getType(e,tacit),
                       getName(),
                       encodeDefVal ? getDefaultValue() : 0,
                       getExplicit());
}

types::ty *formal::getType(coenv &e, bool tacit) {
  types::ty *t = start ? start->getType(base->trans(e), e, tacit)
    : base->trans(e, tacit);
  if (t->kind == ty_void && !tacit) {
    em->compiler(getPos());
    *em << "can't declare parameters of type void";
    return primError();
  }
  return t;
}
  
void formals::prettyprint(ostream &out, int indent)
{
  prettyname(out, "formals",indent);

  for(list<formal *>::iterator p = fields.begin(); p != fields.end(); ++p)
    (*p)->prettyprint(out, indent+1);
}

void formals::addToSignature(signature& sig,
                             coenv &e, bool encodeDefVal, bool tacit) {
  for(list<formal *>::iterator p = fields.begin(); p != fields.end(); ++p)
    sig.add((*p)->trans(e, encodeDefVal, tacit));

  if (rest)
    sig.addRest(rest->trans(e, encodeDefVal, tacit));
}

// Returns the types of each parameter as a signature.
// encodeDefVal means that it will also encode information regarding
// the default values into the signature
signature *formals::getSignature(coenv &e, bool encodeDefVal, bool tacit)
{
  signature *sig = new signature;
  addToSignature(*sig,e,encodeDefVal,tacit);
  return sig;
}


// Returns the corresponding function type, assuming it has a return
// value of types::ty *result.
function *formals::getType(types::ty *result, coenv &e,
                           bool encodeDefVal,
			   bool tacit)
{
  function *ft = new function(result);
  addToSignature(ft->sig,e,encodeDefVal,tacit);
  return ft;
}

void formal::transAsVar(coenv &e, int index) {
  symbol *name = getName();
  if (name) {
    trans::access *a = e.c.accessFormal(index);
    assert(a);

    // Suppress error messages because they will already be reported
    // when the formals are translated to yield the type earlier.
    types::ty *t = getType(e, true);
    varEntry *v = new varEntry(t, a);

    e.e.addVar(getPos(), name, v);
  }
}

void formals::trans(coenv &e)
{
  int index = 0;

  for (list<formal *>::iterator p=fields.begin(); p!=fields.end(); ++p) {
    (*p)->transAsVar(e, index);
    ++index;
  }

  if (rest) {
    rest->transAsVar(e, index);
    ++index;
  }
}

void formals::reportDefaults()
{
  for(list<formal *>::iterator p = fields.begin(); p != fields.end(); ++p)
    if ((*p)->reportDefault())
      return;
  
  if (rest)
    rest->reportDefault();
}

void fundef::prettyprint(ostream &out, int indent)
{
  result->prettyprint(out, indent+1);
  params->prettyprint(out, indent+1);
  body->prettyprint(out, indent+1);
}

function *fundef::transType(coenv &e, bool tacit) {
  return params->getType(result->trans(e, tacit), e, tacit);
}

types::ty *fundef::trans(coenv &e) {
  function *ft=transType(e, false);
  
  // Create a new function environment.
  coder fc = e.c.newFunction(ft);
  coenv fe(fc,e.e);

  // Translate the function.
  fe.e.beginScope();
  params->trans(fe);
  
  body->trans(fe);

  types::ty *rt = ft->result;
  if (rt->kind != ty_void &&
      rt->kind != ty_error &&
      !body->returns()) {
    em->error(body->getPos());
    *em << "function must return a value";
  }

  fe.e.endScope();

  // Put an instance of the new function on the stack.
  vm::lambda *l = fe.c.close();
  e.c.encode(inst::pushclosure);
  e.c.encode(inst::makefunc, l);

  return ft;
}

void fundec::prettyprint(ostream &out, int indent)
{
  prettyindent(out, indent);
  out << "fundec '" << *id << "'\n";

  fun.prettyprint(out, indent);
}

function *fundec::opType(function *f)
{
  function *ft = new function(primBoolean());
  ft->add(f);
  ft->add(f);

  return ft;
}

void fundec::addOps(coenv &e, function *f)
{
  function *ft = opType(f);
  e.e.addVar(getPos(), symbol::trans("=="),
      new varEntry(ft, new bltinAccess(run::boolFuncEq)));
  e.e.addVar(getPos(), symbol::trans("!="),
      new varEntry(ft, new bltinAccess(run::boolFuncNeq)));
}

void fundec::trans(coenv &e)
{
  transAsField(e,0);
}

void fundec::transAsField(coenv &e, record *r)
{
  function *ft = fun.transType(e, true);
  assert(ft);

  addOps(e,ft);
  
  addVar(getPos(), e, r, id, ft, &fun);
} 

} // namespace absyntax
