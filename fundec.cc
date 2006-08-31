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

varinit *Default=new definit(position::nullPos());
  
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
                       encodeDefVal ? (bool) getDefaultValue() : 0,
                       getExplicit());
}

types::ty *formal::getType(coenv &e, bool tacit) {
  types::ty *bt = base->trans(e, tacit);
  types::ty *t = start ? start->getType(bt, e, tacit) : bt;
  if (t->kind == ty_void && !tacit) {
    em->compiler(getPos());
    *em << "cannot declare parameters of type void";
    return primError();
  }

  return t;
}
  
void formal::addOps(coenv &e) {
  if (start)
    start->addOps(base->trans(e, true), e);
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

// Another helper class. Does an assignment, but relying only on the
// destination for the type.
class basicAssignExp : public exp {
  exp *dest;
  varinit *value;
public:
  basicAssignExp(position pos, exp *dest, varinit *value) 
    : exp(pos), dest(dest), value(value) {}

  types::ty *getType(coenv &e) {
    return dest->getType(e);
  }

  types::ty *trans(coenv &e) {
    // This doesn't handle overloaded types for the destination.
    value->transToType(e, getType(e));
    dest->transWrite(e, getType(e));
    return getType(e);
  }
};
  
void transDefault(coenv &e, position pos, varEntry *v, varinit *init) {
  // This roughly translates into the expression
  //   if (isDefault(x))
  //     x=init;
  // where x is the variable in v and isDefault is a function that tests
  // whether x is the default argument token.
  varEntryExp vee(pos, v);
  ifStm is(pos,
           new callExp(pos,
               new varEntryExp(pos,
                   new function(primBoolean(), v->getType()),
                   run::isDefault),
               &vee),
           new expStm(pos,
               new basicAssignExp(pos,
                   &vee,
                   init)));
  is.trans(e);                                        
}

void formal::transAsVar(coenv &e, int index) {
  symbol *name = getName();
  if (name) {
    trans::access *a = e.c.accessFormal(index);
    assert(a);

    // Suppress error messages because they will already be reported
    // when the formals are translated to yield the type earlier.
    types::ty *t = getType(e, true);
    varEntry *v = new varEntry(t, a, 0);

    // Add operations if a new array type is formed.  This will add to the
    // environment at the start of the function, when the formals are also added
    // to the environment.
    addOps(e);

    // Translate the default argument before adding the formal to the
    // environment, consistent with the initializers of variables.
    if (defval)
      transDefault(e, getPos(), v, defval);

    e.e.addVar(name, v);
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

void fundec::trans(coenv &e)
{
  transAsField(e,0);
}

void fundec::transAsField(coenv &e, record *r)
{
  function *ft = fun.transType(e, true);
  assert(ft);

  e.e.addFunctionOps(ft);
  if (r)
    r->e.addFunctionOps(ft);
  
  createVar(getPos(), e, r, id, ft, &fun);
} 

} // namespace absyntax
