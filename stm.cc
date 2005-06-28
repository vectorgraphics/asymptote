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
#include "coenv.h"
#include "stm.h"

namespace absyntax {

using namespace trans;
using namespace types;

void stm::prettyprint(ostream &out, int indent)
{
  prettyname(out,"stm",indent);
}

void stm::trans(coenv &)
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

void blockStm::trans(coenv &e)
{
  e.e.beginScope();
  for (list<runnable *>::iterator p = stms.begin(); p != stms.end(); ++p) {
    (*p)->markTrans(e);
  }
  e.e.endScope();
}

void blockStm::transAsRecordBody(coenv &e, record *r)
{
  e.e.beginScope();
  for (list<runnable *>::iterator p = stms.begin(); p != stms.end(); ++p) {
    (*p)->markTransAsField(e, r);
  }
  e.e.endScope();

  // Put record into finished state.
  e.c.encode(inst::pushclosure);
  e.c.close();
}

void expStm::prettyprint(ostream &out, int indent)
{
  prettyname(out,"expStm",indent);

  body->prettyprint(out, indent+1);
}

void expStm::trans(coenv &e)
{
  types::ty_kind kind = body->trans(e)->kind;
  if (kind != types::ty_void &&
      kind != types::ty_void)
    // Remove any value it puts on the stack.
    e.c.encode(inst::pop);
}


void ifStm::prettyprint(ostream &out, int indent)
{
  prettyname(out,"ifStm",indent);

  test->prettyprint(out, indent+1);
  onTrue->prettyprint(out, indent+1);
  if (onFalse)
    onFalse->prettyprint(out, indent+1);
}

void ifStm::trans(coenv &e)
{
  test->transToType(e, types::primBoolean());

  int elseLabel = e.c.fwdLabel();
  int end = e.c.fwdLabel();

  e.c.useLabel(inst::njmp,elseLabel);

  onTrue->markTrans(e);
  e.c.useLabel(inst::jmp,end);
  
  e.c.defLabel(elseLabel);
  // Produces efficient code whether or not there is an else clause.
  if (onFalse)
    onFalse->markTrans(e);

  e.c.defLabel(end);
}


void whileStm::prettyprint(ostream &out, int indent)
{
  prettyname(out,"whileStm",indent);

  test->prettyprint(out, indent+1);
  body->prettyprint(out, indent+1);
}

void whileStm::trans(coenv &e)
{
  int start = e.c.defLabel();
  e.c.pushContinue(start);
  test->transToType(e, types::primBoolean());

  int end = e.c.fwdLabel();
  e.c.pushBreak(end);
  e.c.useLabel(inst::njmp,end);

  body->markTrans(e);

  e.c.useLabel(inst::jmp,start);
  e.c.defLabel(end);

  e.c.popBreak();
  e.c.popContinue();
}


void doStm::prettyprint(ostream &out, int indent)
{
  prettyname(out,"doStm",indent);

  body->prettyprint(out, indent+1);
  test->prettyprint(out, indent+1);
}

void doStm::trans(coenv &e)
{
  int testLabel = e.c.fwdLabel();
  e.c.pushContinue(testLabel);
  int end = e.c.fwdLabel();
  e.c.pushBreak(end);
 
  int start = e.c.defLabel();

  body->markTrans(e);  
  
  e.c.defLabel(testLabel);
  test->transToType(e, types::primBoolean());
  e.c.useLabel(inst::cjmp,start);
  e.c.defLabel(end);

  e.c.popBreak();
  e.c.popContinue();
}


void forStm::prettyprint(ostream &out, int indent)
{
  prettyname(out,"forStm",indent);

  if (init) init->prettyprint(out, indent+1);
  if (test) test->prettyprint(out, indent+1);
  if (update) update->prettyprint(out, indent+1);
  body->prettyprint(out, indent+1);
}

void forStm::trans(coenv &e)
{
  // Any vardec in the initializer needs its own scope.
  e.e.beginScope();
  if(init) init->markTrans(e);

  int ctarget = e.c.fwdLabel();
  e.c.pushContinue(ctarget);
  int end = e.c.fwdLabel();
  e.c.pushBreak(end);

  int start = e.c.defLabel();
  if(test) {
    test->transToType(e, types::primBoolean());
    e.c.useLabel(inst::njmp,end);
  }

  body->markTrans(e);

  e.c.defLabel(ctarget);
  
  if (update) update->markTrans(e);
  e.c.useLabel(inst::jmp,start);

  e.c.defLabel(end);

  e.e.endScope();
  e.c.popBreak();
  e.c.popContinue();
}


void breakStm::prettyprint(ostream &out, int indent)
{
  prettyname(out,"breakStm",indent);
}

void breakStm::trans(coenv &e)
{
  if (!e.c.encodeBreak()) {
    em->error(getPos());
    *em << "break statement outside of a loop";
  }
}


void continueStm::prettyprint(ostream &out, int indent)
{
  prettyname(out,"continueStm",indent);
}

void continueStm::trans(coenv &e)
{
  if (!e.c.encodeContinue()) {
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

void returnStm::trans(coenv &e)
{
  types::ty *t = e.c.getReturnType();

  if (t->kind == ty_void) {
    if (value) {
      em->error(getPos());
      *em << "function cannot return a value";
    }
    if (e.c.isRecord())
      e.c.encode(inst::pushclosure);
  }
  else {
    if (value) {
      value->transToType(e, t);
    }
    else {
      em->error(getPos());
      *em << "function must return a value";
    }
  }

  // NOTE: Currently, a return statement in a module definition will end
  // the initializer.  Should this be allowed?
  e.c.encode(inst::ret);
}


void stmExpList::prettyprint(ostream &out, int indent)
{
  prettyname(out, "stmExpList",indent);

  for (list<stm *>::iterator p = stms.begin(); p != stms.end(); ++p)
    (*p)->prettyprint(out, indent+1);
}

void stmExpList::trans(coenv &e)
{
  for (list<stm *>::iterator p = stms.begin(); p != stms.end(); ++p)
    (*p)->markTrans(e);
}


} // namespace absyntax
