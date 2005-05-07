/*****
 * exp.cc
 * andy hammerlindl 2002/8/19
 *
 * represents the abstract syntax tree for the expressions in the
 * language.  this is translated into virtual machine code using trans()
 * and with the aid of the environment class.
 *****/

#include "exp.h"
#include "errormsg.h"
#include "runtime.h"
#include "coenv.h"
#include "dec.h"
#include "stm.h"
#include "camp.tab.h"  // For the binary operator names

namespace absyntax {

using namespace types;
using namespace trans;

void exp::transCall(coenv &e, types::ty *target)
{
    trans(e, target);
    e.c.encode(inst::popcall);
}

void arrayinit::prettyprint(ostream &out, int indent)
{
  prettyname(out, "arrayinit",indent);

  for (list<varinit *>::iterator p = inits.begin(); p != inits.end(); ++p)
    (*p)->prettyprint(out, indent+1);
}

void arrayinit::trans(coenv &e, types::ty *target)
{
  types::ty *celltype;
  if (target->kind != types::ty_array) {
    em->error(getPos());
    *em << "array initializer used for non-array";
    celltype = types::primError();
  }
  else {
    celltype = ((types::array *)target)->celltype;
  }
  
  // Push the values on the stack.
  for (list<varinit *>::iterator p = inits.begin(); p != inits.end(); ++p)
    (*p)->trans(e, celltype);

  // Push the number of cells and call the array maker.
  e.c.encode(inst::intpush,(int)inits.size());
  e.c.encode(inst::builtin, run::newInitializedArray);
}


void exp::prettyprint(ostream &out, int indent)
{
  prettyname(out, "exp",indent);
}

void exp::trans(coenv &e, types::ty *target)
{
  // Defer to context-less translation.
  e.implicitCast(getPos(), target, trans(e));
}


void nameExp::prettyprint(ostream &out, int indent)
{
  prettyname(out, "nameExp",indent);

  value->prettyprint(out, indent+1);
}


void fieldExp::prettyprint(ostream &out, int indent)
{
  prettyindent(out, indent);
  out << "fieldExp '" << *field << "'\n";

  object->prettyprint(out, indent+1);
}

record *fieldExp::getRecord(types::ty *t)
{
  if (t->kind == ty_overloaded) {
    t = ((overloaded *)t)->resolve(0);
    if (!t) {
      return 0;
    }
  }

  switch(t->kind) {
    case ty_record:
      return (record *)t;
    case ty_error:
      return 0;
    default:
      return 0;
  } 
}

types::ty *fieldExp::getObject(coenv& e)
{
  types::ty *t = object->getType(e);
  if (t->kind == ty_overloaded) {
    t=((overloaded *)t)->resolve(0);
    if(!t) return primError();
  }
  return t;
}

record *fieldExp::transRecord(coenv &e, types::ty *t)
{
  object->trans(e, t);

  switch(t->kind) {
    case ty_record:
      return (record *)t;
    case ty_error:
      return 0;
    default:
      em->error(object->getPos());
      *em << "type '" << *t << "' is not a structure";
      return 0;
  }
}

void fieldExp::trans(coenv &e, types::ty *target)
{
  types::ty *ot = getObject(e);
#if 0
  if (ot->kind == ty_error) {
    em->error(getPos());
    *em << "expression is not a structure";
    return;
  }
#endif
  
  varEntry *v = ot->virtualField(field, target->getSignature());
  if (v) {
    // Push object onto stack.
    object->trans(e, ot);

    // Call instead of reading as it is a virtual field.
    v->getLocation()->encodeCall(getPos(), e.c);
    e.implicitCast(getPos(), target, v->getType());
    return;
  }

  record *r = transRecord(e, ot);
  if (!r)
    return;

  v = r->lookupExactVar(field, target->getSignature());

  if (v) {
    access *loc = v->getLocation();
    loc->encodeRead(getPos(), e.c, r->getLevel());
    e.implicitCast(getPos(), target, v->getType());
  }
  else {
    em->error(getPos());
    *em << "no matching field of name '" << *field
        << "' in type '" << *r << "'";
  }
}

types::ty *fieldExp::trans(coenv &e)
{
  trans::ty *t = getType(e);
  if (!t) {
    em->error(getPos());
    *em << "no matching field of name '" << *field
        << "' in type '"  << *getObject(e) << "'";
    return primError();
  }
  else if (t->kind == ty_overloaded) {
    em->error(getPos());
    *em << "use of field '" << *field << "' is ambiguous"
        << " in type '" << *getObject(e) << "'";
    return primError();
  }
  else {
    trans(e, t);
    return t;
  }
}

types::ty *fieldExp::getType(coenv &e)
{
  types::ty *ot = getObject(e);
  if (ot->kind == ty_error) return primError();

  types::ty *vt = ot->virtualFieldGetType(field);
  if (vt)
    return vt;

  record *r = getRecord(ot);
  if (r) {
    types::ty *t = r->varGetType(field);
    return t ? t : primError();
  }
  else
    return primError();
}

void fieldExp::transWrite(coenv &e, types::ty *target)
{
  types::ty *ot = getObject(e);

  // Look for virtual fields.
  varEntry *v = ot->virtualField(field, target->getSignature());
  if (v) {
    // Push qualifier onto stack.
    object->trans(e, ot);
    
    em->error(getPos());
    *em << "virtual field '" << *field << "' of '" << *ot
        << "' cannot be modified";
  }
 
  record *r = transRecord(e, ot);
  if (!r)
    return;

  v = r->lookupExactVar(field, target->getSignature());

  if (v) {
    access *loc = v->getLocation();
    loc->encodeWrite(getPos(), e.c, r->getLevel());
  }
  else {
    em->error(getPos());
    *em << "no matching field of name '" << *field
        << "' in type  '" << *r << "'";
  }
}

void fieldExp::transCall(coenv &e, types::ty *target)
{
  types::ty *ot = object->getType(e);

  // Look for virtual fields.
  varEntry *v = ot->virtualField(field, target->getSignature());
  if (v) {
    // Push object onto stack.
    object->trans(e, ot);
    
    // Call instead of reading as it is a virtual field.
    v->getLocation()->encodeCall(getPos(), e.c);
    e.implicitCast(getPos(), target, v->getType());

    // In this case, the virtual field will construct a vm::func object
    // and push it on the stack.
    // Then, pop and call the function.
    e.c.encode(inst::popcall);
    return;
  }

  record *r = transRecord(e, ot);
  if (!r)
    return;

  v = r->lookupExactVar(field, target->getSignature());

  if (v) {
    access *loc = v->getLocation();
    loc->encodeCall(getPos(), e.c, r->getLevel());
  }
  else {
    em->error(getPos());
    *em << "no matching field of name '" << *field
        << "' in type  '" << *r << "'";
  }
}


void subscriptExp::prettyprint(ostream &out, int indent)
{
  prettyindent(out, indent);
  out << "subscriptExp\n";

  set->prettyprint(out, indent+1);
  index->prettyprint(out, indent+1);
}

array *subscriptExp::getArrayType(coenv &e)
{
  types::ty *a = set->getType(e);
  if (a->kind == ty_overloaded) {
    a = ((overloaded *)a)->resolve(0);
    if (!a)
      return 0;
  }

  switch (a->kind) {
    case ty_array:
      return (array *)a;
    case ty_error:
      return 0;
    default:
      return 0;
  }
}

array *subscriptExp::transArray(coenv &e)
{
  types::ty *a = set->getType(e);
  if (a->kind == ty_overloaded) {
    a = ((overloaded *)a)->resolve(0);
    if (!a) {
      em->error(set->getPos());
      *em << "expression is not an array";
      return 0;
    }
  }

  set->trans(e, a);

  switch (a->kind) {
    case ty_array:
      return (array *)a;
    case ty_error:
      return 0;
    default:
      em->error(set->getPos());
      *em << "expression is not an array";
      return 0;
  }
}

types::ty *subscriptExp::trans(coenv &e)
{
  array *a = transArray(e);
  if (!a)
    return primError();

  types::ty *t=index->getType(e);
  if(t->kind == ty_array) {
    index->trans(e, types::intArray());
    e.c.encode(inst::builtin, run::arrayIntArray);
    return getArrayType(e);
  }
     
  index->trans(e, types::primInt());
  if(a->celltype->kind == ty_array)
    e.c.encode(inst::builtin, run::arrayArrayRead);
  else e.c.encode(inst::builtin, run::arrayRead);

  return a->celltype;
}

types::ty *subscriptExp::getType(coenv &e)
{
  types::ty *t=index->getType(e);
  array *a = getArrayType(e);
  
  if(t->kind == ty_array)
    return a ? a : primError();
  
  return a ? a->celltype : primError();
}
     
void subscriptExp::transWrite(coenv &e, types::ty *t)
{
  array *a = transArray(e);
  if (!a)
    return;

  e.implicitCast(getPos(), a->celltype, t);

  index->trans(e, types::primInt());
  e.c.encode(inst::builtin, run::arrayWrite);
}


void thisExp::prettyprint(ostream &out, int indent)
{
  prettyname(out, "thisExp", indent);
}

types::ty *thisExp::trans(coenv &e)
{
  if (!e.c.encodeThis()) {
    em->error(getPos());
    *em << "static use of 'this' expression";
  }
  return getType(e);
}

types::ty *thisExp::getType(coenv &e)
{
  return e.c.thisType();
}

void scaleExp::prettyprint(ostream &out, int indent)
{
  prettyname(out, "scaleExp",indent);
  left->prettyprint(out, indent+1);
  right->prettyprint(out, indent+1);
}

types::ty *scaleExp::trans(coenv &e)
{
  types::ty *lt = left->getType(e);
  if (lt->kind != types::ty_int && lt->kind != types::ty_real) {
    if (lt->kind != types::ty_error) {
      em->error(left->getPos());
      *em << "only numeric constants can do implicit scaling";
    }
    right->trans(e);
    return types::primError();
  }

  if (!right->scalable()) {
    em->warning(right->getPos());
    *em << "implicit scaling may be unintentional";
  }

  // Defer to the binaryExp for multiplication.
  binaryExp b(getPos(), left, symbol::trans("*"), right);
  return b.trans(e);
}

types::ty *scaleExp::getType(coenv &e)
{
  // Defer to the binaryExp for multiplication.
  binaryExp b(getPos(), left, symbol::trans("*"), right);
  return b.getType(e);
}


void intExp::prettyprint(ostream &out, int indent)
{
  prettyindent(out,indent);
  out << "intExp: " << value << "\n";
}

types::ty *intExp::trans(coenv &e)
{
  e.c.encode(inst::intpush,value);
  
  return types::primInt();  
}


void realExp::prettyprint(ostream &out, int indent)
{
  prettyindent(out, indent);
  out << "realExp: " << value << "\n";
}

types::ty *realExp::trans(coenv &e)
{
  e.c.encode(inst::constpush,(item)value);
  
  return types::primReal();  
}

void stringExp::prettyprint(ostream &out, int indent)
{
  prettyindent(out, indent);
  out << "stringExp '" << str << "'\n";
}

types::ty *stringExp::trans(coenv &e)
{
  e.c.encode(inst::constpush,(item)str);
  
  return types::primString();  
}


void booleanExp::prettyprint(ostream &out, int indent)
{
  prettyindent(out, indent);
  out << "booleanExp: " << value << "\n";
}

types::ty *booleanExp::trans(coenv &e)
{
  e.c.encode(inst::constpush,(item)value);
  
  return types::primBoolean();  
}

void nullPictureExp::prettyprint(ostream &out, int indent)
{
  prettyname(out, "nullPictureExp",indent);
}

types::ty *nullPictureExp::trans(coenv &e)
{
  e.c.encode(inst::builtin, run::nullFrame);
  
  return types::primPicture();  
}

void nullPathExp::prettyprint(ostream &out, int indent)
{
  prettyname(out, "nullPathExp",indent);
}

types::ty *nullPathExp::trans(coenv &e)
{
  e.c.encode(inst::builtin, run::nullPath);
  
  return types::primPath();  
}

void nullExp::prettyprint(ostream &out, int indent)
{
  prettyname(out, "nullExp",indent);
}

types::ty *nullExp::trans(coenv &)
{
  // Things get put on the stack when ty_null
  // is cast to an appropriate type
  return types::primNull();  
}

void explist::prettyprint(ostream &out, int indent)
{
  prettyname(out, "explist",indent);
  for (expvector::iterator p = exps.begin();
       p != exps.end(); ++p)
    (*p)->prettyprint(out, indent+1);
}

types::ty *explist::trans(coenv &e, int index)
{
  assert((unsigned)index < exps.size());
  return exps[index]->trans(e);
}

void explist::trans(coenv &e, types::ty *target, int index)
{
  assert((unsigned)index < exps.size());
  exps[index]->trans(e, target);
}

types::ty *explist::getType(coenv &e, int index)
{
  assert((unsigned)index < exps.size());
  return exps[index]->getType(e);
}


void callExp::prettyprint(ostream &out, int indent)
{
  prettyname(out, "callExp",indent);

  callee->prettyprint(out, indent+1);
  args->prettyprint(out, indent+1);
}

signature *callExp::argTypes(coenv &e)
{
  signature *sig=new signature;

  size_t n = args->size();
  for (size_t i = 0; i < n; i++) {
    types::ty *t = args->getType(e,(int) i);
    if (t->kind == types::ty_error)
      return 0;
    sig->add(t);
  }

  return sig;
}

types::ty *callExp::trans(coenv &e)
{
  // First figure out the signature of what we want to call.
  signature *sig=argTypes(e);

  if (!sig) {
    // Cycle through the parameters to report all errors.
    // NOTE: This may report inappropriate ambiguity errors. 
    for (size_t i = 0; i < args->size(); i++) {
      args->trans(e,(int) i);
    }
    return types::primError();
  }

  // Figure out what function types we can call.
  trans::ty *ft = callee->getType(e);
  if (ft->kind == ty_error) {
    return callee->trans(e);
  }
  if (ft->kind == ty_overloaded) {
    ft = ((overloaded *)ft)->resolve(sig);
  }

  if (ft == 0) {
    em->error(getPos());
    symbol *s = callee->getName();
    if (s)
      *em << "no matching function \'" << *s << *sig << "\'";
    else
      *em << "no matching function for signature \'" << *sig << "\'";
    return primError();
  }
  else if (ft->kind == ty_overloaded) {
    em->error(getPos());
    *em << "call with signature \'" << *sig << "\' is ambiguous";
    return primError();
  }
  else if (ft->kind != ty_function) {
    em->error(getPos());
    symbol *s = callee->getName();
    if (s)
      *em << "\'" << *s << "\' is not a function";
    else
      *em << "called expression is not a function";
    return primError();
  }
  else if (!castable(ft->getSignature(), sig)) {
    em->error(getPos());
    const char *separator=ft->getSignature()->getNumFormals() > 1 ? "\n" : " ";
    symbol *s=callee->getName();
    *em << "cannot call" << separator << "'" 
	<< *((function *) ft)->getResult() << " ";
    if(s) *em << *s;
    *em << *ft->getSignature() << "'" << separator;
    switch(sig->getNumFormals()) {
      case 0:
        *em << "without parameters";
        break;
      case 1:
        *em << "with parameter '" << *sig << "'";
        break;
      default:
        *em << "with parameters\n'" << *sig << "'";
    }
    return primError();
  }

  // We have a winner.
  signature *real_sig = ft->getSignature();
  assert(real_sig);

  int m = real_sig->getNumFormals();
  int n = (int) args->size();
 
  // Put the arguments on the stack. 
  for (int i = 0, j = 0; i < m; i++) {
    if (j < n && (real_sig->getExplicit(i) ? 
		  equivalent(real_sig->getFormal(i), args->getType(e,j)) :
		  castable(real_sig->getFormal(i), args->getType(e,j)))) {
      // Argument given
      args->trans(e, real_sig->getFormal(i), j);
      j++;
    } else { // Use default value instead
      varinit *def=real_sig->getDefault(i);
      assert(def);
      def->trans(e, real_sig->getFormal(i));
    }
  }

  // Translate the call.
  callee->transCall(e, ft);

  return ((function *)ft)->result;
}

types::ty *callExp::getType(coenv &e)
{
  // First figure out the signature of what we want to call.
  signature *sig=argTypes(e);
  if (!sig) {
    return types::primError();
  }

  // Figure out what function types we can call.
  trans::ty *ft = callee->getType(e);
  if (ft->kind == ty_error) {
    return primError();
  }
  if (ft->kind == ty_overloaded) {
    ft = ((overloaded *)ft)->resolve(sig);
  }

  if (ft == 0 || ft->kind != ty_function) {
    return primError();
  }

  // We have a winner.
  return ((function *)ft)->result;
}
    
void pairExp::prettyprint(ostream &out, int indent)
{
  prettyname(out, "pairExp",indent);

  x->prettyprint(out, indent+1);
  y->prettyprint(out, indent+1);
}

types::ty *pairExp::trans(coenv &e)
{
  types::ty *xt = x->trans(e);
  e.implicitCast(x->getPos(), types::primReal(), xt);
  
  types::ty *yt = y->trans(e);
  e.implicitCast(y->getPos(), types::primReal(), yt);

  e.c.encode(inst::builtin, run::realRealToPair);

  return types::primPair();
}


void dimensions::prettyprint(ostream &out, int indent)
{
  prettyindent(out, indent);
  out << "dimensions (" << depth << ")\n";
}

types::ty *dimensions::truetype(types::ty *base)
{
  if (base->kind == ty_void) {
    em->compiler(getPos());
    *em << "can't declare array of type void";
    return primError();
  }
  for (size_t d = depth; d > 0; d--) {
    base = new types::array(base);
  }
  return base;
}


void castExp::prettyprint(ostream &out, int indent)
{
  prettyname(out, "castExp",indent);

  target->prettyprint(out, indent+1);
  castee->prettyprint(out, indent+1);
}

types::ty *castExp::trans(coenv &e)
{
  types::ty *t = target->typeTrans(e);

  types::ty *source = castee->getType(e);

  // Find the source type to actually use.
  types::ty *intermed = types::explicitCastType(t, source);
  if (intermed == 0) {
    em->error(getPos());
    *em << "cannot cast '" << *source << "' to '" << *target << "'"; 
    return primError();
  }
  else if (intermed->kind == ty_overloaded) {
    // NOTE: I can't see how this situation could arise.
    em->error(getPos());
    *em << "cast is ambiguous";
  }

  castee->trans(e, intermed);
  e.explicitCast(getPos(), t, intermed);

  return t;
}

types::ty *castExp::getType(coenv &e)
{
  return target->typeTrans(e, true);
}

#if 0
void binaryExp::prettyprint(ostream &out, int indent)
{
  prettyindent(out, indent);
  out << "binaryExp '" << *op << "'\n";

  left->prettyprint(out, indent+1);
  right->prettyprint(out, indent+1);
}

types::ty *binaryExp::trans(coenv &e)
{
  /* The conditional && and || need jumps in the translated code in
   * order to conditionally evaluate their operands (not done for arrays).
   */
  types::ty *t1 = left->getType(e), *t2 = right->getType(e);
  if (t1->kind != ty_array && t2->kind != ty_array) {
    if (op == symbol::trans("&&")) {
      left->trans(e, primBoolean());

      int second = e.c.fwdLabel();
      e.c.useLabel(inst::cjmp,second);
      e.c.encode(inst::constpush,(item)false);

      int end = e.c.fwdLabel();
      e.c.useLabel(inst::jmp,end);
    
      e.c.defLabel(second);

      right->trans(e, primBoolean());

      e.c.defLabel(end);
      return types::primBoolean();
    }
    else if (op == symbol::trans("||")) {
      left->trans(e, primBoolean());

      int pushtrue = e.c.fwdLabel();
      e.c.useLabel(inst::cjmp,pushtrue);

      right->trans(e, primBoolean());

      int end = e.c.fwdLabel();
      e.c.useLabel(inst::jmp,end);

      e.c.defLabel(pushtrue);
      e.c.encode(inst::constpush,(item)true);

      e.c.defLabel(end);
      return types::primBoolean();
    }
  }
  
  if (t1->kind == ty_error ||
      t2->kind == ty_error)
  {
    left->trans(e);
    right->trans(e);
    return primError();
  }
  
  signature sig;
  sig.add(t1);
  sig.add(t2);

  // Figure out what function types we have for this operator.
  trans::ty *ft = e.e.varGetType(op);
  if (ft == 0) {
    em->error(getPos());
    *em << "no matching operation \'" << *op << "\'";
    return primError();
  }
  if (ft->kind == ty_error) {
    return primError();
  }
  if (ft->kind == ty_overloaded) {
    // NOTE: Change to "promoteResolve" or something for operators.
    ft = ((overloaded *)ft)->resolve(&sig);
  }

  if (ft == 0) {
    em->error(getPos());
    *em << "no matching operation \'" << *op << "\'";
    return primError();
  }
  else if (ft->kind == ty_overloaded) {
    em->error(getPos());
    *em << "operation \'" << *op << "\' is ambiguous";
    return primError();
  }
  else if (!castable(ft->getSignature(), &sig)) {
    em->error(getPos());
    *em << "no operation \'" << *t1 << "\' \'" << *op << "\' \'" << *t2 
	<< "\'";
    return primError();
  }

  // We have a winner.
  signature *real_sig = ft->getSignature();
  assert(real_sig);

  if (real_sig->getNumFormals() != 2) {
    em->compiler(getPos());
    *em << "default values used with operator";
    return primError();
  }
 
  // Put the arguments on the stack. 
  left->trans(e, real_sig->getFormal(0));
  right->trans(e, real_sig->getFormal(1));

  // Call the operator.
  varEntry *v = e.e.lookupExactVar(op, real_sig);
  v->getLocation()->encodeCall(getPos(), e.c);

  return ((function *)ft)->result;
}

types::ty *binaryExp::getType(coenv &e)
{
  /* The conditional && and || need jumps in the translated code in
   * order to conditionally evaluate their operands (not done for arrays).
   */
  types::ty *t1 = left->getType(e), *t2 = right->getType(e);
  if ((op == symbol::trans("&&") ||
       op == symbol::trans("||")) && 
      t1->kind != ty_array && t2->kind != ty_array) {
    return primBoolean();
  }

  if (t1->kind == ty_error ||
      t2->kind == ty_error)
  {
    return primError();
  }
  
  signature sig;
  sig.add(t1);
  sig.add(t2);

  // Figure out what function types we have for this operator.
  trans::ty *ft = e.e.varGetType(op);
  if (ft == 0 || ft->kind == ty_error) {
    return primError();
  }
  if (ft->kind == ty_overloaded) {
    // NOTE: Change to "promoteResolve" or something for operators.
    ft = ((overloaded *)ft)->resolve(&sig);
  }

  if (ft == 0 || ft->kind == ty_overloaded) {
    return primError();
  }
  else {
    return ((function *)ft)->result;
  }
}
#endif


void conditionalExp::prettyprint(ostream &out, int indent)
{
  prettyname(out, "conditionalExp",indent);

  test->prettyprint(out, indent+1);
  onTrue->prettyprint(out, indent+1);
  onFalse->prettyprint(out, indent+1);
}

void conditionalExp::trans(coenv &e, types::ty *target)
{
  types::ty *t=test->getType(e);
  if(t->kind == ty_array && ((array *)t)->celltype == primBoolean()) {
    if(target->kind == ty_array) {
      test->trans(e, types::boolArray());
      onTrue->trans(e, target);
      onFalse->trans(e, target);
      e.c.encode(inst::builtin, run::arrayConditional);
      return;
    }
  }
  
  test->trans(e, types::primBoolean());

  int tlabel = e.c.fwdLabel();
  e.c.useLabel(inst::cjmp,tlabel);

  onFalse->trans(e, target);

  int end = e.c.fwdLabel();
  e.c.useLabel(inst::jmp,end);

  e.c.defLabel(tlabel);
  onTrue->trans(e, target);

  e.c.defLabel(end);
}

types::ty *conditionalExp::trans(coenv &e)
{
  types::ty *t = promote(onTrue->getType(e), onFalse->getType(e));
  if (!t) {
    em->error(getPos());
    *em << "types in conditional expression do not match";
    return primError();
  }
  else if (t->kind == ty_overloaded) {
    em->error(getPos());
    *em << "type of conditional expression is ambiguous";
    return primError();
  }

  trans(e,t);
  return t;
}

types::ty *conditionalExp::getType(coenv &e)
{
  types::ty *t = promote(onTrue->getType(e), onFalse->getType(e));
  return t ? t : primError();
}
 

// Checks if the expression can be translated as an array.
bool isAnArray(exp *x, coenv &e)
{
  types::ty *t=x->getType(e);
  if (t->kind == ty_overloaded)
    t=dynamic_cast<overloaded *>(t)->resolve(0);
  return t && t->kind==ty_array;
}

types::ty *andOrExp::trans(coenv &e)
{
  if (isAnArray(left,e) || isAnArray(right,e)) {
    binaryExp be(getPos(), left, op, right);
    return be.trans(e);
  }
  else
    return baseTrans(e);
}

types::ty *andOrExp::getType(coenv &e)
{
  if (isAnArray(left,e) || isAnArray(right,e)) {
    binaryExp be(getPos(), left, op, right);
    return be.getType(e);
  }
  else
    return baseGetType(e);
}

void orExp::prettyprint(ostream &out, int indent)
{
  prettyname(out, "orExp", indent);

  left->prettyprint(out, indent+1);
  right->prettyprint(out, indent+1);
}

types::ty *orExp::baseTrans(coenv &e)
{
  booleanExp be(pos, true);
  conditionalExp ce(pos, left, &be, right);
  ce.trans(e, primBoolean());

  return baseGetType(e);
}


void andExp::prettyprint(ostream &out, int indent)
{
  prettyname(out, "andExp", indent);

  left->prettyprint(out, indent+1);
  right->prettyprint(out, indent+1);
}

types::ty *andExp::baseTrans(coenv &e)
{
  booleanExp be(pos, false);
  conditionalExp ce(pos, left, right, &be);
  ce.trans(e, primBoolean());

  return getType(e);
}


void joinExp::guidearray::prettyprint(ostream &out, int indent)
{
  prettyname(out, "guidearray", indent);
  base.prettyprint(out, indent+1);
}

void joinExp::prettyprint(ostream &out, int indent)
{
  prettyindent(out,indent);
  out << "joinExp '" << *op << "'\n";
  guides.prettyprint(out, indent+1);
}

types::ty *joinExp::trans(coenv& e)
{
  // Translate as a unary operator converting the guide array to a single guide.
  unaryExp u(getPos(),&guides,op);
  return u.trans(e);
}

types::ty *joinExp::getType(coenv& e)
{
  // Translate as a unary operator converting the guide array to a single guide.
  unaryExp u(getPos(),&guides,op);
  return u.getType(e);
}


void specExp::prettyprint(ostream &out, int indent)
{
  prettyindent(out,indent);
  out << "specExp '" << *op << "' " 
      << (s==camp::OUT ? "out" :
          s==camp::IN  ? "in" :
                         "invalid side") << endl;

  arg->prettyprint(out, indent+1);
}

types::ty *specExp::trans(coenv &e)
{
  intExp ie(getPos(), (int)s);
  binaryExp be(getPos(), arg, op, &ie);
  return be.trans(e);
}

types::ty *specExp::getType(coenv &e)
{
  intExp ie(getPos(), (int)s);
  binaryExp be(getPos(), arg, op, &ie);
  return be.getType(e);
}

void assignExp::prettyprint(ostream &out, int indent)
{
  prettyname(out, "assignExp",indent);

  dest->prettyprint(out, indent+1);
  value->prettyprint(out, indent+1);
}

void assignExp::trans(coenv &e, types::ty *target)
{
  value->trans(e, target);
  dest->transWrite(e, target);
}

types::ty *assignExp::trans(coenv &e)
{
  types::ty *lt = dest->getType(e), *rt = value->getType(e);

  if (lt->kind == ty_error)
    return dest->trans(e);
  if (rt->kind == ty_error)
    return value->trans(e);

  types::ty *t = castType(lt, rt);
  if (!t) {
    em->error(getPos());
    *em << "cannot convert '" << *rt << "' to '" << *lt << "' in assignment";
    return primError();
  }
  else if (t->kind == ty_overloaded) {
    em->error(getPos());
    *em << "assignment is ambiguous";
    return primError();
  }
  else {
    trans(e, t);
    return t;
  }
}

types::ty *assignExp::getType(coenv &e)
{
  types::ty *lt = dest->getType(e), *rt = value->getType(e);
  types::ty *t = castType(lt, rt);

  return t ? t : primError();
}

void selfExp::prettyprint(ostream &out, int indent)
{
  prettyindent(out, indent);
  out << "selfExp '" << *op << "'\n";

  dest->prettyprint(out, indent+1);
  value->prettyprint(out, indent+1);
}

types::ty *selfExp::trans(coenv &e)
{
  // Convert into the operation and the assign.
  // NOTE: This can cause multiple evaluations.
  binaryExp be(getPos(), dest, op, value);
  assignExp ae(getPos(), dest, &be);
  return ae.trans(e);
}

types::ty *selfExp::getType(coenv &e)
{
  // Convert into the operation and the assign.
  binaryExp be(getPos(), dest, op, value);
  assignExp ae(getPos(), dest, &be);
  return ae.getType(e);
}

void prefixExp::prettyprint(ostream &out, int indent)
{
  prettyindent(out, indent);
  out << "prefixExp '" << *op << "'\n";
  
  dest->prettyprint(out, indent+1);
}

types::ty *prefixExp::trans(coenv &e)
{
  // Convert into the operation and the assign.
  // NOTE: This can cause multiple evaluations.
  intExp ie(getPos(), 1);
  binaryExp be(getPos(), dest, op, &ie);
  assignExp ae(getPos(), dest, &be);

  return ae.trans(e);
}

types::ty *prefixExp::getType(coenv &e)
{
  // Convert into the operation and the assign.
  intExp ie(getPos(), 1);
  binaryExp be(getPos(), dest, op, &ie);
  assignExp ae(getPos(), dest, &be);

  return ae.getType(e);
}

void postfixExp::prettyprint(ostream &out, int indent)
{
  prettyindent(out, indent);
  out << "postfixExp <illegal>";
  out << "postfixExp <illegal> '" << *op << "'\n";

  dest->prettyprint(out, indent+1);
}

types::ty *postfixExp::trans(coenv &)
{
  em->error(getPos());
  *em << "postfix expressions are not allowed";
  return primError();
}


} // namespace absyntax

