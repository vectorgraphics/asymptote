/*****
 * stm.cc
 * Andy Hammerlindl 2002/8/30
 *
 * Statements are everything in the language that do something on their
 * own.  Statements are different from declarations in that statements
 * do not modify the environment.  Translation of a statement puts the
 * stack code to run it into the instruction stream.
 *****/

#include <fstream>
#include "errormsg.h"
#include "settings.h"

#include "stm.h"

namespace as {

using namespace trans;
using namespace types;

void stm::prettyprint(ostream &out, int indent)
{
  prettyname(out,"stm",indent);
}

void stm::trans(env &)
{
  em->compiler(getPos());
  *em <<  "base stm in abstract syntax";
}


void emptyStm::prettyprint(ostream &out, int indent)
{
  prettyname(out,"emptyStm",indent);
}


void blockStm::prettystms(ostream &out, int indent)
{
  for (list<runnable *>::iterator p = stms.begin(); p != stms.end(); ++p)
    (*p)->prettyprint(out, indent+1);
}

void blockStm::prettyprint(ostream &out, int indent)
{
  prettyname(out,"blockStm",indent);
  prettystms(out, indent+1);
}

void blockStm::trans(env &e)
{
  e.beginScope();
  for (list<runnable *>::iterator p = stms.begin(); p != stms.end(); ++p) {
    (*p)->markTrans(e);
  }
  e.endScope();
}

void blockStm::transAsRecordBody(env &e, record *r)
{
  e.beginScope();
  for (list<runnable *>::iterator p = stms.begin(); p != stms.end(); ++p) {
    (*p)->markTransAsField(e, r);
  }
  e.endScope();

  // Put record into finished state.
  e.close();
  r->close();
}

void file::prettyprint(ostream &out, int indent)
{
  prettyname(out,"file",indent);
  prettystms(out, indent+1);
}

void file::transAsRecordBody(env &e, record *r)
{
  blockStm::transAsRecordBody(e,r);
}


void expStm::prettyprint(ostream &out, int indent)
{
  prettyname(out,"expStm",indent);

  body->prettyprint(out, indent+1);
}

void expStm::trans(env &e)
{
  if(!body->stmable()) {
    em->warning(getPos());
    *em << "expression (in whole or part) without side-effects";
  }

  types::ty_kind kind = body->trans(e)->kind;
  if (kind != types::ty_void &&
      kind != types::ty_void)
    // Remove any value it puts on the stack.
    e.encode(inst::pop);
}


void ifStm::prettyprint(ostream &out, int indent)
{
  prettyname(out,"ifStm",indent);

  test->prettyprint(out, indent+1);
  onTrue->prettyprint(out, indent+1);
  if (onFalse)
    onFalse->prettyprint(out, indent+1);
}

void ifStm::trans(env &e)
{
  test->trans(e, types::primBoolean());

  int elseLabel = e.fwdLabel();
  int end = e.fwdLabel();

  e.encode(inst::njmp);
  e.useLabel(elseLabel);

  onTrue->markTrans(e);
  e.encode(inst::jmp);
  e.useLabel(end);
  
  e.defLabel(elseLabel);
  // Produces efficient code whether or not there is an else clause.
  if (onFalse)
    onFalse->markTrans(e);

  e.defLabel(end);
}


void whileStm::prettyprint(ostream &out, int indent)
{
  prettyname(out,"whileStm",indent);

  test->prettyprint(out, indent+1);
  body->prettyprint(out, indent+1);
}

void whileStm::trans(env &e)
{
  int start = e.defLabel();
  e.pushContinue(start);
  test->trans(e, types::primBoolean());

  int end = e.fwdLabel();
  e.pushBreak(end);
  e.encode(inst::njmp);
  e.useLabel(end);

  body->markTrans(e);

  e.encode(inst::jmp);
  e.useLabel(start);
  e.defLabel(end);

  e.popBreak();
  e.popContinue();
}


void doStm::prettyprint(ostream &out, int indent)
{
  prettyname(out,"doStm",indent);

  body->prettyprint(out, indent+1);
  test->prettyprint(out, indent+1);
}

void doStm::trans(env &e)
{
  int testLabel = e.fwdLabel();
  e.pushContinue(testLabel);
  int end = e.fwdLabel();
  e.pushBreak(end);
 
  int start = e.defLabel();

  body->markTrans(e);  
  
  e.defLabel(testLabel);
  test->trans(e, types::primBoolean());
  e.encode(inst::cjmp);
  e.useLabel(start);
  e.defLabel(end);

  e.popBreak();
  e.popContinue();
}


void forStm::prettyprint(ostream &out, int indent)
{
  prettyname(out,"forStm",indent);

  if (init) init->prettyprint(out, indent+1);
  if (test) test->prettyprint(out, indent+1);
  if (update) update->prettyprint(out, indent+1);
  body->prettyprint(out, indent+1);
}

void forStm::trans(env &e)
{
  // Any vardec in the initializer needs its own scope.
  e.beginScope();
  if(init) init->markTrans(e);

  int ctarget = e.fwdLabel();
  e.pushContinue(ctarget);
  int end = e.fwdLabel();
  e.pushBreak(end);

  int start = e.defLabel();
  if(test) {
    test->trans(e, types::primBoolean());
    e.encode(inst::njmp);
    e.useLabel(end);
  }

  body->markTrans(e);

  e.defLabel(ctarget);
  
  if (update) update->markTrans(e);
  e.encode(inst::jmp);
  e.useLabel(start);

  e.defLabel(end);

  e.endScope();
  e.popBreak();
  e.popContinue();
}


void breakStm::prettyprint(ostream &out, int indent)
{
  prettyname(out,"breakStm",indent);
}

void breakStm::trans(env &e)
{
  if (!e.encodeBreak()) {
    em->error(getPos());
    *em << "break statement outside of a loop";
  }
}


void continueStm::prettyprint(ostream &out, int indent)
{
  prettyname(out,"continueStm",indent);
}

void continueStm::trans(env &e)
{
  if (!e.encodeContinue()) {
    em->error(getPos()); 
    *em << "continue statement outside of a loop";
  }
}


void returnStm::prettyprint(ostream &out, int indent)
{
  prettyname(out, "returnStm",indent);

  if (value)
    value->prettyprint(out, indent+1);
}

void returnStm::trans(env &e)
{
  types::ty *t = e.getReturnType();

  if (t->kind == ty_void) {
    if (value) {
      em->error(getPos());
      *em << "function cannot return a value";
    }
  }
  else {
    if (value) {
      value->trans(e, t);
    }
    else {
      em->error(getPos());
      *em << "function must return a value";
    }
  }

  // NOTE: Currently, a return statement in a module definition will end
  // the initializer.  Should this be allowed?
  e.encode(inst::ret);
}


void stmExpList::prettyprint(ostream &out, int indent)
{
  prettyname(out, "stmExpList",indent);

  for (list<stm *>::iterator p = stms.begin(); p != stms.end(); ++p)
    (*p)->prettyprint(out, indent+1);
}

void stmExpList::trans(env &e)
{
  for (list<stm *>::iterator p = stms.begin(); p != stms.end(); ++p)
    (*p)->markTrans(e);
}


} // namespace as
