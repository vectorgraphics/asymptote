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

#include "dec.h"
#include "stm.h"
#include "camp.tab.h"  // For the binary operator names

namespace as {

using namespace types;
using namespace trans;

vector<varinit *> defaultExp;


void arrayinit::prettyprint(ostream &out, int indent)
{
  prettyname(out, "arrayinit",indent);

  for (list<varinit *>::iterator p = inits.begin(); p != inits.end(); ++p)
    (*p)->prettyprint(out, indent+1);
}

void arrayinit::trans(env &e, types::ty *target)
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
  e.encode(inst::intpush);
  e.encode((int)inits.size());
  e.encode(inst::builtin);
  e.encode(run::newInitializedArray);
}


void exp::prettyprint(ostream &out, int indent)
{
  prettyname(out, "exp",indent);
}

void exp::trans(env &e, types::ty *target)
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

types::ty *fieldExp::getObject(env& e)
{
  types::ty *t = object->getType(e);
  if (t->kind == ty_overloaded) {
    t=((overloaded *)t)->resolve(0);
    if(!t) return primError();
  }
  return t;
}

record *fieldExp::transRecord(env &e, types::ty *t)
{
  object->trans(e, t);

  switch(t->kind) {
    case ty_record:
      return (record *)t;
    case ty_error:
      return 0;
    default:
      em->error(object->getPos());
      *em << "type '" << *t << "' is not a record";
      return 0;
  }
}

void fieldExp::trans(env &e, types::ty *target)
{
  types::ty *ot = getObject(e);
#if 0
  if (ot->kind == ty_error) {
    em->error(getPos());
    *em << "expression is not a record";
    return;
  }
#endif
  
  varEntry *v = ot->virtualField(field, target->getSignature());
  if (v) {
    // Push object onto stack.
    object->trans(e, ot);

    // Call instead of reading as it is a virtual field.
    v->getLocation()->encodeCall(getPos(), e);
    e.implicitCast(getPos(), target, v->getType());
    return;
  }

  record *r = transRecord(e, ot);
  if (!r)
    return;

  v = r->lookupExactVar(field, target->getSignature());

  if (v) {
    access *loc = v->getLocation();
    loc->encodeRead(getPos(), e, r->getLevel());
    e.implicitCast(getPos(), target, v->getType());
  }
  else {
    em->error(getPos());
    *em << "no matching field of name '" << *field
        << "' in type '" << *r << "'";
  }
}

types::ty *fieldExp::trans(env &e)
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

types::ty *fieldExp::getType(env &e)
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

void fieldExp::transWrite(env &e, types::ty *target)
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
    loc->encodeWrite(getPos(), e, r->getLevel());
  }
  else {
    em->error(getPos());
    *em << "no matching field of name '" << *field
        << "' in type  '" << *r << "'";
  }
}

void fieldExp::transCall(env &e, types::ty *target)
{
  types::ty *ot = object->getType(e);

  // Look for virtual fields.
  varEntry *v = ot->virtualField(field, target->getSignature());
  if (v) {
    // Push object onto stack.
    object->trans(e, ot);
    
    // Call instead of reading as it is a virtual field.
    v->getLocation()->encodeCall(getPos(), e);
    e.implicitCast(getPos(), target, v->getType());

    // In this case, the virtual field will construct a vm::func object
    // and push it on the stack.
    // Then, pop and call the function.
    e.encode(inst::popcall);
    return;
  }

  record *r = transRecord(e, ot);
  if (!r)
    return;

  v = r->lookupExactVar(field, target->getSignature());

  if (v) {
    access *loc = v->getLocation();
    loc->encodeCall(getPos(), e, r->getLevel());
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

array *subscriptExp::getArrayType(env &e)
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

array *subscriptExp::transArray(env &e)
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

types::ty *subscriptExp::trans(env &e)
{
  array *a = transArray(e);
  if (!a)
    return primError();

  types::ty *t=index->getType(e);
  if(t->kind == ty_array) {
    index->trans(e, types::intArray());
    e.encode(inst::builtin);
    e.encode(run::arrayIntArray);
    return getArrayType(e);
  }
     
  index->trans(e, types::primInt());
  e.encode(inst::builtin);
  if(a->celltype->kind == ty_array)
    e.encode(run::arrayArrayRead);
  else e.encode(run::arrayRead);

  return a->celltype;
}

types::ty *subscriptExp::getType(env &e)
{
  types::ty *t=index->getType(e);
  array *a = getArrayType(e);
  
  if(t->kind == ty_array)
    return a ? a : primError();
  
  return a ? a->celltype : primError();
}
     
void subscriptExp::transWrite(env &e, types::ty *t)
{
  array *a = transArray(e);
  if (!a)
    return;

  e.implicitCast(getPos(), a->celltype, t);

  index->trans(e, types::primInt());
  e.encode(inst::builtin);
  e.encode(run::arrayWrite);
}


void thisExp::prettyprint(ostream &out, int indent)
{
  prettyname(out, "thisExp", indent);
}

types::ty *thisExp::trans(env &e)
{
  if (!e.encodeThis()) {
    em->error(getPos());
    *em << "static use of 'this' expression";
  }
  return getType(e);
}

types::ty *thisExp::getType(env &e)
{
  return e.thisType();
}

void scaleExp::prettyprint(ostream &out, int indent)
{
  prettyname(out, "scaleExp",indent);
  left->prettyprint(out, indent+1);
  right->prettyprint(out, indent+1);
}

types::ty *scaleExp::trans(env &e)
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
    types::ty *rt = right->trans(e);
    if (rt->kind != types::ty_error) {
      em->error(right->getPos());
      *em << "expression cannot be implicitly scaled";
    }
    return types::primError();
  }

  // Defer to the binaryExp for multiplication.
  // NOTE: This may cause problems later for memory deletion.
  binaryExp b(getPos(), left, symbol::trans("*"), right);
  return b.trans(e);
}

types::ty *scaleExp::getType(env &e)
{
  // Defer to the binaryExp for multiplication.
  // NOTE: This may cause problems later for memory deletion.
  binaryExp b(getPos(), left, symbol::trans("*"), right);
  return b.getType(e);
}


void intExp::prettyprint(ostream &out, int indent)
{
  prettyindent(out,indent);
  out << "intExp: " << value << "\n";
}

types::ty *intExp::trans(env &e)
{
  e.encode(inst::intpush);
  e.encode(value);
  
  return types::primInt();  
}


void realExp::prettyprint(ostream &out, int indent)
{
  prettyindent(out, indent);
  out << "realExp: " << value << "\n";
}

types::ty *realExp::trans(env &e)
{
  e.encode(inst::constpush);
  e.encode((item)value);
  
  return types::primReal();  
}

void stringExp::prettyprint(ostream &out, int indent)
{
  prettyindent(out, indent);
  out << "stringExp '" << str << "'\n";
}

types::ty *stringExp::trans(env &e)
{
  e.encode(inst::constpush);
  e.encode((item)str);
  
  return types::primString();  
}


void booleanExp::prettyprint(ostream &out, int indent)
{
  prettyindent(out, indent);
  out << "booleanExp: " << value << "\n";
}

types::ty *booleanExp::trans(env &e)
{
  e.encode(inst::constpush);
  e.encode((item)value);
  
  return types::primBoolean();  
}

void nullPictureExp::prettyprint(ostream &out, int indent)
{
  prettyname(out, "nullPictureExp",indent);
}

types::ty *nullPictureExp::trans(env &e)
{
  e.encode(inst::builtin);
  e.encode(run::nullFrame);
  
  return types::primPicture();  
}

void nullPathExp::prettyprint(ostream &out, int indent)
{
  prettyname(out, "nullPathExp",indent);
}

types::ty *nullPathExp::trans(env &e)
{
  e.encode(inst::builtin);
  e.encode(run::nullPath);
  
  return types::primPath();  
}

void nullExp::prettyprint(ostream &out, int indent)
{
  prettyname(out, "nullExp",indent);
}

types::ty *nullExp::trans(env &)
{
  // Things get put on the stack when ty_null
  // is cast to an appropriate type
  return types::primNull();  
}

void explist::prettyprint(ostream &out, int indent)
{
  prettyname(out, "explist",indent);
  for (std::vector<exp *>::iterator p = exps.begin(); p != exps.end(); ++p)
    (*p)->prettyprint(out, indent+1);
}

types::ty *explist::trans(env &e, int index)
{
  assert((unsigned)index < exps.size());
  return exps[index]->trans(e);
}

void explist::trans(env &e, types::ty *target, int index)
{
  assert((unsigned)index < exps.size());
  exps[index]->trans(e, target);
}

types::ty *explist::getType(env &e, int index)
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

types::ty *callExp::trans(env &e)
{
  // First figure out the signature of what we want to call.
  signature sig;

  int n = (int) args->size();
  int anyErrors = false;
  for (int i = 0; i < n; i++) {
    types::ty *t = args->getType(e, i);
    if (t->kind == types::ty_error) {
      anyErrors = true;
      break;
    }
    sig.add(t);
  }
  if (anyErrors) {
    // Cycle through the parameters to report all errors.
    // NOTE: This may report inappropriate ambiguity errors. 
    for (int i = 0; i < n; i++) {
      args->trans(e, i);
    }
    return types::primError();
  }

  // Figure out what function types we can call.
  trans::ty *ft = callee->getType(e);
  if (ft->kind == ty_error) {
    return callee->trans(e);
  }
  if (ft->kind == ty_overloaded) {
    ft = ((overloaded *)ft)->resolve(&sig);
  }

  if (ft == 0) {
    em->error(getPos());
    symbol *s = callee->getName();
    if (s)
      *em << "no matching function \'" << *s << sig << "\'";
    else
      *em << "no matching function for signature \'" << sig << "\'";
    return primError();
  }
  else if (ft->kind == ty_overloaded) {
    em->error(getPos());
    *em << "call with signature \'" << sig << "\' is ambiguous";
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
  else if (!castable(ft->getSignature(), &sig)) {
    em->error(getPos());
    *em << "cannot call type '" << *ft << "' with";
    switch(sig.getNumFormals()) {
      case 0:
        *em << "out parameters";
        break;
      case 1:
        *em << " parameter '" << sig << "'";
        break;
      default:
        *em << " parameters '" << sig << "'";
    }
    return primError();
  }

  // We have a winner.
  signature *real_sig = ft->getSignature();
  assert(real_sig);

  int m = real_sig->getNumFormals();
 
  // Put the arguments on the stack. 
  for (int i = 0, j = 0; i < m; i++) {
    if(j < n && castable(real_sig->getFormal(i), args->getType(e,j))) {
      // Argument given
      args->trans(e, real_sig->getFormal(i), j);
      j++;
    } else { // Use default value instead
      size_t k=real_sig->getDefault(i);
      assert(k);
      assert(k <= defaultExp.size());
      defaultExp[k-1]->trans(e, real_sig->getFormal(i));
    }
  }

  // Translate the call.
  callee->transCall(e, ft);

  return ((function *)ft)->result;
}

types::ty *callExp::getType(env &e)
{
  // First figure out the signature of what we want to call.
  signature sig;

  int n = (int) args->size();
  int anyErrors = false;
  for (int i = 0; i < n; i++) {
    types::ty *t = args->getType(e, i);
    if (t->kind == types::ty_error) {
      anyErrors = true;
      break;
    }
    sig.add(t);
  }
  if (anyErrors) {
    return types::primError();
  }

  // Figure out what function types we can call.
  trans::ty *ft = callee->getType(e);
  if (ft->kind == ty_error) {
    return primError();
  }
  if (ft->kind == ty_overloaded) {
    ft = ((overloaded *)ft)->resolve(&sig);
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

types::ty *pairExp::trans(env &e)
{
  types::ty *xt = x->trans(e);
  e.implicitCast(x->getPos(), types::primReal(), xt);
  
  types::ty *yt = y->trans(e);
  e.implicitCast(y->getPos(), types::primReal(), yt);

  e.encode(inst::builtin);
  e.encode(run::realRealToPair);

  return types::primPair();
}

void unaryExp::prettyprint(ostream &out, int indent)
{
  prettyindent(out, indent);
  out << "unaryExp '" << *op << "'\n";

  base->prettyprint(out, indent+1);
}

types::ty *unaryExp::trans(env &e)
{
  types::ty *t = base->getType(e);
  if (t->kind == ty_error) {
    base->trans(e);
    return primError();
  }
  
  signature sig;
  sig.add(t);

  // Figure out what function types we have for this operator.
  trans::ty *ft = e.varGetType(op);
  if (ft == 0) {
    em->error(getPos());
    *em << "no matching unary operation \'" << *op << "\'";
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
    *em << "no matching unary operation \'" << *op << "\'";
    return primError();
  }
  else if (ft->kind == ty_overloaded) {
    em->error(getPos());
    *em << "unary operation \'" << *op << "\' is ambiguous";
    return primError();
  }
  else if (!castable(ft->getSignature(), &sig)) {
    em->error(getPos());
    *em << "no unary operation \'" << *op << "\' \'" << *t << "\'";
    return primError();
  }

  // We have a winner.
  signature *real_sig = ft->getSignature();
  assert(real_sig);

  if (real_sig->getNumFormals() != 1) {
    em->compiler(getPos());
    *em << "default values used with unary operator";
    return primError();
  }
 
  // Put the argument on the stack. 
  base->trans(e, real_sig->getFormal(0));

  // Call the operator.
  varEntry *v = e.lookupExactVar(op, real_sig);
  v->getLocation()->encodeCall(getPos(), e);

  return ((function *)ft)->result;
}

types::ty *unaryExp::getType(env &e)
{
  types::ty *t = base->getType(e);
  if (t->kind == ty_error) 
  {
    return primError();
  }
  
  signature sig;
  sig.add(t);

  // Figure out what function types we have for this operator.
  trans::ty *ft = e.varGetType(op);
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

types::ty *castExp::trans(env &e)
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

types::ty *castExp::getType(env &e)
{
  return target->typeTrans(e, true);
}


void binaryExp::prettyprint(ostream &out, int indent)
{
  prettyindent(out, indent);
  out << "binaryExp '" << *op << "'\n";

  left->prettyprint(out, indent+1);
  right->prettyprint(out, indent+1);
}

types::ty *binaryExp::trans(env &e)
{
  /* The conditional && and || need jumps in the translated code in
   * order to conditionally evaluate their operands (not done for arrays).
   */
  types::ty *t1 = left->getType(e), *t2 = right->getType(e);
  if (t1->kind != ty_array && t2->kind != ty_array) {
    if (op == symbol::trans("&&")) {
      left->trans(e, primBoolean());

      int second = e.fwdLabel();
      e.encode(inst::cjmp);
      e.useLabel(second);
      e.encode(inst::constpush);
      e.encode((item)false);

      int end = e.fwdLabel();
      e.encode(inst::jmp);
      e.useLabel(end);
    
      e.defLabel(second);

      right->trans(e, primBoolean());

      e.defLabel(end);
      return types::primBoolean();
    }
    else if (op == symbol::trans("||")) {
      left->trans(e, primBoolean());

      int pushtrue = e.fwdLabel();
      e.encode(inst::cjmp);
      e.useLabel(pushtrue);

      right->trans(e, primBoolean());

      int end = e.fwdLabel();
      e.encode(inst::jmp);
      e.useLabel(end);

      e.defLabel(pushtrue);
      e.encode(inst::constpush);
      e.encode((item)true);

      e.defLabel(end);
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
  trans::ty *ft = e.varGetType(op);
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
  varEntry *v = e.lookupExactVar(op, real_sig);
  v->getLocation()->encodeCall(getPos(), e);

  return ((function *)ft)->result;
}

types::ty *binaryExp::getType(env &e)
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
  trans::ty *ft = e.varGetType(op);
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


void conditionalExp::prettyprint(ostream &out, int indent)
{
  prettyname(out, "conditionalExp",indent);

  test->prettyprint(out, indent+1);
  onTrue->prettyprint(out, indent+1);
  onFalse->prettyprint(out, indent+1);
}

void conditionalExp::trans(env &e, types::ty *target)
{
  types::ty *t=test->getType(e);
  if(t->kind == ty_array && ((array *)t)->celltype == primBoolean()) {
    if(target->kind == ty_array) {
      test->trans(e, types::boolArray());
      onTrue->trans(e, target);
      onFalse->trans(e, target);
      e.encode(inst::builtin);
      e.encode(run::arrayConditional);
      return;
    }
  }
  
  test->trans(e, types::primBoolean());

  int tlabel = e.fwdLabel();
  e.encode(inst::cjmp);
  e.useLabel(tlabel);

  onFalse->trans(e, target);

  int end = e.fwdLabel();
  e.encode(inst::jmp);
  e.useLabel(end);

  e.defLabel(tlabel);
  onTrue->trans(e, target);

  e.defLabel(end);
}

types::ty *conditionalExp::trans(env &e)
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

types::ty  *conditionalExp::getType(env &e)
{
  types::ty *t = promote(onTrue->getType(e), onFalse->getType(e));
  return t ? t : primError();
}
 

void givenDir::prettyprint(ostream &out, int indent)
{
  prettyname(out, "givenDir",indent);

  base->prettyprint(out, indent+1);
}

void givenDir::trans(env &e)
{
  e.implicitCast(getPos(), types::primPair(), base->trans(e));
}


void curlDir::prettyprint(ostream &out, int indent)
{
  prettyname(out, "curlDir",indent);

  base->prettyprint(out, indent+1);
}

void curlDir::trans(env &e)
{
  e.implicitCast(getPos(), types::primReal(), base->trans(e));
}


void join::prettyprint(ostream &out, int indent)
{
  prettyindent(out, indent);
  out << "join";

  // Add tension and atleast line if they are used.
  if (leftCont || rightCont) {
    if (tension) {
      if (atleast)
        out << " (tension atleast)";
      else
        out << " (tension)";
    }
    else {
      out << " (controls)";
    }
  }
  out << "\n";

  if (leftDir)
    leftDir->prettyprint(out, indent+1);
  if (leftCont)
    leftCont->prettyprint(out, indent+1);
  if (rightCont)
    rightCont->prettyprint(out, indent+1);
  if (rightDir)
    rightDir->prettyprint(out, indent+1);
}

void join::trans(env &e)
{
  int flags = run::NULL_JOIN;

  if (leftDir) {
    leftDir->trans(e);
    flags |= leftDir->leftFlags();
  }

  if (leftCont) {
     if (tension) {
       if (atleast)
	 flags |= run::TENSION_ATLEAST;
       e.implicitCast(leftCont->getPos(), types::primReal(),
	               leftCont->trans(e));
       flags |= run::LEFT_TENSION;

       if (rightCont) {
         e.implicitCast(rightCont->getPos(), types::primReal(),
	                 rightCont->trans(e));
         flags |= run::RIGHT_TENSION;
       }
     }
     else { // controls
       e.implicitCast(leftCont->getPos(), types::primPair(),
	               leftCont->trans(e));
       flags |= run::LEFT_CONTROL;

       if (rightCont) {
         e.implicitCast(rightCont->getPos(), types::primPair(),
	                 rightCont->trans(e));
         flags |= run::RIGHT_CONTROL;
       }
     }
  }

  if (rightDir) {
    rightDir->trans(e);
    flags |= rightDir->rightFlags();
  }

  // Tell the join function whats been put on the stack.
  e.encode(inst::intpush);
  e.encode(flags);
}


void joinExp::prettyprint(ostream &out, int indent)
{
  prettyname(out, "joinExp",indent);

  left->prettyprint(out, indent+1);
  middle->prettyprint(out, indent+1);
  right->prettyprint(out, indent+1);
}

types::ty *joinExp::trans(env &e)
{
  e.implicitCast(left->getPos(), types::primGuide(), left->trans(e));
  middle->trans(e);
  e.implicitCast(right->getPos(), types::primGuide(), right->trans(e));

  e.encode(inst::builtin);
  e.encode(run::newJoin);

  return types::primGuide();
}


void cycleExp::prettyprint(ostream &out, int indent)
{
  prettyname(out, "cycleExp",indent);
}

types::ty *cycleExp::trans(env &e)
{
  e.encode(inst::builtin);
  e.encode(run::newCycle);

  return types::primGuide();
}

void dirguideExp::prettyprint(ostream &out, int indent)
{
  prettyname(out, "dirguideExp",indent);

  base->prettyprint(out, indent+1);
  tag->prettyprint(out, indent+1);
}

types::ty *dirguideExp::trans(env &e)
{
  e.implicitCast(base->getPos(), types::primGuide(), base->trans(e));
  tag->trans(e);
  
  // Tell the dirtag function what type of dirtag it has.
  e.encode(inst::intpush);
  e.encode(tag->rightFlags());
  e.encode(inst::builtin);
  e.encode(run::newDirguide);

  return types::primGuide();
}


void assignExp::prettyprint(ostream &out, int indent)
{
  prettyname(out, "assignExp",indent);

  dest->prettyprint(out, indent+1);
  value->prettyprint(out, indent+1);
}

void assignExp::trans(env &e, types::ty *target)
{
  value->trans(e, target);
  dest->transWrite(e, target);
}

types::ty *assignExp::trans(env &e)
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

types::ty *assignExp::getType(env &e)
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

types::ty *selfExp::trans(env &e)
{
  // Convert into the operation and the assign.
  // NOTE: This can cause multiple evaluations.
  binaryExp be(getPos(), dest, op, value);
  assignExp ae(getPos(), dest, &be);
  return ae.trans(e);
}

types::ty *selfExp::getType(env &e)
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

types::ty *prefixExp::trans(env &e)
{
  // Convert into the operation and the assign.
  // NOTE: This can cause multiple evaluations.
  intExp ie(getPos(), 1);
  binaryExp be(getPos(), dest, op, &ie);
  assignExp ae(getPos(), dest, &be);

  return ae.trans(e);
}

types::ty *prefixExp::getType(env &e)
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

types::ty *postfixExp::trans(env &)
{
  em->error(getPos());
  *em << "postfix expressions are not allowed";
  return primError();
}


} // namespace as

