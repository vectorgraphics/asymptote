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
#include "exp.h"
#include "stm.h"

namespace absyntax {

using namespace trans;
using namespace types;

void stm::prettyprint(ostream &out, Int indent)
{
  prettyname(out,"stm",indent);
}


void emptyStm::prettyprint(ostream &out, Int indent)
{
  prettyname(out,"emptyStm",indent);
}


void blockStm::prettyprint(ostream &out, Int indent)
{
  prettyname(out,"blockStm",indent);

  base->prettyprint(out, indent+1);
}


void expStm::prettyprint(ostream &out, Int indent)
{
  prettyname(out,"expStm",indent);

  body->prettyprint(out, indent+1);
}

void expStm::baseTrans(coenv &e, exp *expr)
{
  types::ty_kind kind = expr->trans(e)->kind;
  if (kind != types::ty_void)
    // Remove any value it puts on the stack.
    e.c.encode(inst::pop);
}

void expStm::trans(coenv &e) {
  baseTrans(e, body);
}

exp *tryToWriteExp(coenv &e, exp *body)
{
  // First check if it is the kind of expression that should be written.
  if (body->writtenToPrompt() &&
      settings::getSetting<bool>("interactiveWrite"))
    {
      types::ty *t=body->cgetType(e);
      if (t->kind == ty_error) {
        return body;
      }
      else {
        exp *callee=new nameExp(body->getPos(), symbol::trans("write"));
        exp *call=new callExp(body->getPos(), callee, body);

        types::ty *ct=call->getType(e);
        if (ct->kind == ty_error || ct->kind == ty_overloaded) {
          return body;
        }
        else {
          // Issue a warning if the act of writing turns an ambiguous expression
          // into an unambiguous one.
          if (t->kind == ty_overloaded && settings::warn("writeoverloaded")) {
            em.warning(body->getPos());
            em << "writing overloaded";
            if (body->getName())
              em << " variable '" << *body->getName() << "'";
            else
              em << " expression";
          }
          return call;
        }
      }
    }
  else {
    return body;
  }
}

void expStm::interactiveTrans(coenv &e)
{
  baseTrans(e, tryToWriteExp(e, body));
}


void ifStm::prettyprint(ostream &out, Int indent)
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

  Int elseLabel = e.c.fwdLabel();
  Int end = e.c.fwdLabel();

  e.c.useLabel(inst::njmp,elseLabel);

  onTrue->markTrans(e);
  e.c.useLabel(inst::jmp,end);
  
  e.c.defLabel(elseLabel);
  // Produces efficient code whether or not there is an else clause.
  if (onFalse)
    onFalse->markTrans(e);

  e.c.defLabel(end);
}


void transLoopBody(coenv &e, stm *body) {
  e.c.encodePushFrame();
  body->markTrans(e);
  e.c.encodePopFrame();
}

void whileStm::prettyprint(ostream &out, Int indent)
{
  prettyname(out,"whileStm",indent);

  test->prettyprint(out, indent+1);
  body->prettyprint(out, indent+1);
}

void whileStm::trans(coenv &e)
{
  Int start = e.c.defLabel();
  e.c.pushContinue(start);
  test->transToType(e, types::primBoolean());

  Int end = e.c.fwdLabel();
  e.c.pushBreak(end);
  e.c.useLabel(inst::njmp,end);

  transLoopBody(e,body);

  e.c.useLabel(inst::jmp,start);
  e.c.defLabel(end);

  e.c.popBreak();
  e.c.popContinue();
}


void doStm::prettyprint(ostream &out, Int indent)
{
  prettyname(out,"doStm",indent);

  body->prettyprint(out, indent+1);
  test->prettyprint(out, indent+1);
}

void doStm::trans(coenv &e)
{
  Int testLabel = e.c.fwdLabel();
  e.c.pushContinue(testLabel);
  Int end = e.c.fwdLabel();
  e.c.pushBreak(end);
 
  Int start = e.c.defLabel();

  transLoopBody(e,body);  
  
  e.c.defLabel(testLabel);
  test->transToType(e, types::primBoolean());
  e.c.useLabel(inst::cjmp,start);
  e.c.defLabel(end);

  e.c.popBreak();
  e.c.popContinue();
}


void forStm::prettyprint(ostream &out, Int indent)
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

  Int ctarget = e.c.fwdLabel();
  e.c.pushContinue(ctarget);
  Int end = e.c.fwdLabel();
  e.c.pushBreak(end);

  Int start = e.c.defLabel();
  if(test) {
    test->transToType(e, types::primBoolean());
    e.c.useLabel(inst::njmp,end);
  }

  transLoopBody(e,body);

  e.c.defLabel(ctarget);
  
  if (update) update->markTrans(e);
  e.c.useLabel(inst::jmp,start);

  e.c.defLabel(end);

  e.e.endScope();
  e.c.popBreak();
  e.c.popContinue();
}

void extendedForStm::prettyprint(ostream &out, Int indent)
{
  prettyindent(out, indent);
  out << "extendedForStm: '" << *var << "'\n";

  start->prettyprint(out, indent+1);
  set->prettyprint(out, indent+1);
  body->prettyprint(out, indent+1);
}

void extendedForStm::trans(coenv &e) {
  // Translate into the syntax:
  //
  // start[] a = set;
  // for (int i=0; i < a.length; ++i) {
  //   start var=a[i];
  //   body
  // }

  position pos=getPos();

  // Use gensyms for the variable names so as not to pollute the namespace.
  symbol *a=symbol::gensym("a");
  symbol *i=symbol::gensym("i");

  // start[] a=set;
  arrayTy at(pos, start, new dimensions(pos));
  decid dec1(pos, new decidstart(pos, a), set);
  vardec(pos, &at, &dec1).trans(e);

  // { start var=a[i]; body }
  block b(pos);
  decid dec2(pos, new decidstart(pos, var), 
             new subscriptExp(pos, new nameExp(pos, a),
                              new nameExp(pos, i)));
  b.add(new vardec(pos, start, &dec2));
  b.add(body);



  // for (int i=0; i < a.length; ++i)
  //   <block>
  forStm(pos, new vardec(pos, new tyEntryTy(pos, primInt()),
                         new decid(pos, new decidstart(pos, i),
                                   new intExp(pos, 0))),
         new binaryExp(pos, new nameExp(pos, i),
                       symbol::trans("<"),
                       new nameExp(pos, new qualifiedName(pos, new simpleName(pos, a),
                                                          symbol::trans("length")))),
         new expStm(pos, new prefixExp(pos, new nameExp(pos, i),
                                       symbol::trans("+"))),
         new blockStm(pos, &b)).trans(e);
}
                              

void breakStm::prettyprint(ostream &out, Int indent)
{
  prettyname(out,"breakStm",indent);
}

void breakStm::trans(coenv &e)
{
  // Loop bodies have their own frame to declare variables for each iteration.
  // Pop out of this frame when jumping out of the loop body.
  e.c.encode(inst::popframe);

  if (!e.c.encodeBreak()) {
    em.error(getPos());
    em << "break statement outside of a loop";
  }
}


void continueStm::prettyprint(ostream &out, Int indent)
{
  prettyname(out,"continueStm",indent);
}

void continueStm::trans(coenv &e)
{
  // Loop bodies have their own frame to declare variables for each iteration.
  // Pop out of this frame when jumping out of the loop body.
  e.c.encode(inst::popframe);

  if (!e.c.encodeContinue()) {
    em.error(getPos()); 
    em << "continue statement outside of a loop";
  }
}


void returnStm::prettyprint(ostream &out, Int indent)
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
      em.error(getPos());
      em << "function cannot return a value";
    }
    if (e.c.isRecord())
      e.c.encode(inst::pushclosure);
  }
  else {
    if (value) {
      value->transToType(e, t);
    }
    else {
      em.error(getPos());
      em << "function must return a value";
    }
  }

  // NOTE: Currently, a return statement in a module definition will end
  // the initializer.  Should this be allowed?
  e.c.encode(inst::ret);
}


void stmExpList::prettyprint(ostream &out, Int indent)
{
  prettyname(out, "stmExpList",indent);

  for (mem::list<stm *>::iterator p = stms.begin(); p != stms.end(); ++p)
    (*p)->prettyprint(out, indent+1);
}

void stmExpList::trans(coenv &e)
{
  for (mem::list<stm *>::iterator p = stms.begin(); p != stms.end(); ++p)
    (*p)->markTrans(e);
}


} // namespace absyntax
