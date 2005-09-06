/*****
 * dec.cc
 * Andy Hammerlindl 2002/8/29
 *
 * Represents the abstract syntax tree for declarations in the language.
 * Also included is an abstract syntax for types as they are most often
 * used with declarations.
 *****/

#include "errormsg.h"
#include "coenv.h"
#include "dec.h"
#include "fundec.h"
#include "stm.h"
#include "exp.h"
#include "modifier.h"
#include "runtime.h"

namespace absyntax {

using namespace trans;
using namespace types;

void nameTy::prettyprint(ostream &out, int indent)
{
  prettyname(out, "nameTy",indent);

  id->prettyprint(out, indent+1);
}

types::ty *nameTy::trans(coenv &e, bool tacit)
{
  return id->typeTrans(e, tacit);
}


void arrayTy::prettyprint(ostream &out, int indent)
{
  prettyname(out, "arrayTy",indent);

  cell->prettyprint(out, indent+1);
  dims->prettyprint(out, indent+1);
}

function *arrayTy::opType(types::ty* t)
{
  function *ft = new function(primBoolean());
  ft->add(t);
  ft->add(t);

  return ft;
}

function *arrayTy::arrayType(types::ty* t)
{
  function *ft = new function(t);
  ft->add(t);

  return ft;
}

function *arrayTy::array2Type(types::ty* t)
{
  function *ft = new function(t);
  ft->add(t);
  ft->add(t);

  return ft;
}

function *arrayTy::cellIntType(types::ty* t)
{
  function *ft = new function(t);
  ft->add(primInt());
  
  return ft;
}
  
function *arrayTy::sequenceType(types::ty* t, types::ty *ct)
{
  function *ft = new function(t);
  function *fc = cellIntType(ct);
  ft->add(fc);
  ft->add(primInt());

  return ft;
}

function *arrayTy::cellTypeType(types::ty* t)
{
  function *ft = new function(t);
  ft->add(t);
  
  return ft;
}
  
function *arrayTy::mapType(types::ty* t, types::ty *ct)
{
  function *ft = new function(t);
  function *fc = cellTypeType(ct);
  ft->add(fc);
  ft->add(t);

  return ft;
}

void arrayTy::addOps(coenv &e, types::ty* t, types::ty *ct)
{
  function *ft = opType(t);
  function *ftarray = arrayType(t);
  function *ftarray2 = array2Type(t);
  function *ftsequence = sequenceType(t,ct);
  function *ftmap = mapType(t,ct);
  
  e.e.addVar(getPos(), symbol::trans("alias"),
      new varEntry(ft,new bltinAccess(run::arrayAlias)));

  if(dims->size() == 1) {
    e.e.addVar(getPos(), symbol::trans("copy"),
	       new varEntry(ftarray,new bltinAccess(run::arrayCopy)));
    e.e.addVar(getPos(), symbol::trans("concat"),
	       new varEntry(ftarray2,new bltinAccess(run::arrayConcat)));
    e.e.addVar(getPos(), symbol::trans("sequence"),
	       new varEntry(ftsequence,new bltinAccess(run::arraySequence)));
    e.e.addVar(getPos(), symbol::trans("map"),
	       new varEntry(ftmap,new bltinAccess(run::arrayFunction)));
  }
  if(dims->size() == 2) {
    e.e.addVar(getPos(), symbol::trans("copy"),
	       new varEntry(ftarray,new bltinAccess(run::array2Copy)));
    e.e.addVar(getPos(), symbol::trans("transpose"),
	       new varEntry(ftarray,new bltinAccess(run::array2Transpose)));
  }
}

types::ty *arrayTy::trans(coenv &e, bool tacit)
{
  types::ty *ct = cell->trans(e, tacit);
  assert(ct);

  types::ty *t = dims->truetype(ct);
  assert(t);
  
  addOps(e,t,ct);
  
  return t;
}


void dec::prettyprint(ostream &out, int indent)
{
  prettyname(out, "dec", indent);
}


void modifierList::prettyprint(ostream &out, int indent)
{
  prettyindent(out,indent);
  out << "modifierList (";
  
  for (list<modifier>::iterator p = mods.begin(); p != mods.end(); ++p) {
    if (p != mods.begin())
      out << ", ";
    switch (*p) {
      case EXPLICIT_STATIC:
	out << "static";
	break;
#if 0	
      case EXPLICIT_DYNAMIC:
	out << "dynamic";
	break;
#endif	
      default:
	out << "invalid code";
    }
  }
  
  for (list<permission>::iterator p = perms.begin(); p != perms.end(); ++p) {
    if (p != perms.begin() || !mods.empty())
      out << ", ";
    switch (*p) {
      case PUBLIC:
	out << "public";
	break;
      case PRIVATE:
	out << "private";
	break;
      default:
	out << "invalid code";
    }
  }

  out << ")\n";
}

bool modifierList::staticSet()
{
  return !mods.empty();
}

modifier modifierList::getModifier()
{
  if (mods.size() > 1) {
    em->error(getPos());
    *em << "too many modifiers";
  }

  assert(staticSet());
  return mods.front();
}

permission modifierList::getPermission()
{
  if (perms.size() > 1) {
    em->error(getPos());
    *em << "too many modifiers";
  }

  return perms.empty() ? READONLY : perms.front();
}


void modifiedRunnable::prettyprint(ostream &out, int indent)
{
  prettyname(out, "modifierRunnable",indent);

  mods->prettyprint(out, indent+1);
  body->prettyprint(out, indent+1);
}

void modifiedRunnable::trans(coenv &e)
{
  transAsField(e,0);
}

void modifiedRunnable::transAsField(coenv &e, record *r)
{
  if (mods->staticSet())
    e.c.pushModifier(mods->getModifier());

  permission p = mods->getPermission();
  if (p != READONLY && (!r || !body->allowPermissions())) {
    em->error(pos);
    *em << "invalid permission modifier";
  }
  e.c.setPermission(p);

  if (r)
    body->transAsField(e,r);
  else
    body->trans(e);

  e.c.clearPermission();
  if (mods->staticSet())
    e.c.popModifier();
}


void decidstart::prettyprint(ostream &out, int indent)
{
  prettyindent(out, indent);
  out << "decidstart '" << *id << "'\n";

  if (dims)
    dims->prettyprint(out, indent+1);
}

types::ty *decidstart::getType(types::ty *base, coenv &, bool)
{
  return dims ? dims->truetype(base) : base;
}


void fundecidstart::prettyprint(ostream &out, int indent)
{
  prettyindent(out, indent);
  out << "fundecidstart '" << *id << "'\n";

  if (dims)
    dims->prettyprint(out, indent+1);
  if (params)
    params->prettyprint(out, indent+1);
}

types::ty *fundecidstart::getType(types::ty *base, coenv &e, bool tacit)
{
  types::ty *result = decidstart::getType(base, e, tacit);

  if (params) {
    return params->getType(result, e, true, tacit);
  }
  else {
    types::ty *t = new function(base);
    return t;
  }
}


void decid::prettyprint(ostream &out, int indent)
{
  prettyname(out, "decid",indent);

  start->prettyprint(out, indent+1);
  if (init)
    init->prettyprint(out, indent+1);
}

void decid::trans(coenv &e, types::ty *base)
{
  transAsField(e,0,base);
}

void addVar(position pos, coenv &e, record *r,
            symbol *id, types::ty *t, varinit *init)
{
  // give the field a location.
  access *a = r ? r->allocField(e.c.isStatic()) :
                  e.c.allocLocal();

  varEntry *ent = r ? new varEntry(t, a, e.c.getPermission(), r) :
                      new varEntry(t, a);

  // Add to the record so it can be accessed when qualified; add to the
  // environment so it can be accessed unqualified in the scope of the
  // record definition.
  if (r)
    r->addVar(id, ent);
  e.e.addVar(pos, id, ent);
  
  if (init)
    init->transToType(e, t);
  else {
    definit d(pos);
    d.transToType(e, t);
  }
  
  a->encode(WRITE, pos, e.c);
  e.c.encode(inst::pop);
}

void addVarOutOfOrder(position pos, coenv &e, record *r,
                      symbol *id, types::ty *t, varinit *init)
{
  // give the field a location.
  access *a = r ? r->allocField(e.c.isStatic()) :
                  e.c.allocLocal();

  varEntry *ent = r ? new varEntry(t, a, e.c.getPermission(), r) :
                      new varEntry(t, a);

  if (init)
    init->transToType(e, t);
  else {
    definit d(pos);
    d.transToType(e, t);
  }
  
  // Add to the record so it can be accessed when qualified; add to the
  // environment so it can be accessed unqualified in the scope of the
  // record definition.
  if (r)
    r->addVar(id, ent);
  e.e.addVar(pos, id, ent);
  
  a->encode(WRITE, pos, e.c);
  e.c.encode(inst::pop);
}

void decid::transAsField(coenv &e, record *r, types::ty *base)
{
  types::ty *t = start->getType(base, e);
  assert(t);
  if (t->kind == ty_void) {
    em->compiler(getPos());
    *em << "can't declare variable of type void";
  }

  addVarOutOfOrder(getPos(), e, r, start->getName(), t, init);
}

void decid::transAsTypedef(coenv &e, types::ty *base)
{
  types::ty *t = start->getType(base, e);
  assert(t);

  if (init) {
    em->error(getPos());
    *em << "type definition cannot have initializer";
  }
   
  // Add to type environment.
  e.e.addType(getPos(), start->getName(), t);
}

void decid::transAsTypedefField(coenv &e, types::ty *base, record *r)
{
  types::ty *t = start->getType(base, e);
  assert(t);

  if (init) {
    em->error(getPos());
    *em << "type definition cannot have initializer";
  }
   
  // Add to type to record and environment.
  r->addType(start->getName(), t);
  e.e.addType(getPos(), start->getName(), t);
}


void decidlist::prettyprint(ostream &out, int indent)
{
  prettyname(out, "decidlist",indent);

  for (list<decid *>::iterator p = decs.begin(); p != decs.end(); ++p)
    (*p)->prettyprint(out, indent+1);
}

void decidlist::trans(coenv &e, types::ty *base)
{
  for (list<decid *>::iterator p = decs.begin(); p != decs.end(); ++p)
    (*p)->trans(e, base);
}

void decidlist::transAsField(coenv &e, record *r, types::ty *base)
{
  for (list<decid *>::iterator p = decs.begin(); p != decs.end(); ++p)
    (*p)->transAsField(e, r, base);
}

void decidlist::transAsTypedef(coenv &e, types::ty *base)
{
  for (list<decid *>::iterator p = decs.begin(); p != decs.end(); ++p)
    (*p)->transAsTypedef(e, base);
}

void decidlist::transAsTypedefField(coenv &e, types::ty *base, record *r)
{
  for (list<decid *>::iterator p = decs.begin(); p != decs.end(); ++p)
    (*p)->transAsTypedefField(e, base, r);
}


void vardec::prettyprint(ostream &out, int indent)
{
  prettyname(out, "vardec",indent);

  base->prettyprint(out, indent+1);
  decs->prettyprint(out, indent+1);
}

void vardec::transAsTypedef(coenv &e)
{
  decs->transAsTypedef(e, base->trans(e));
}

void vardec::transAsTypedefField(coenv &e, record *r)
{
  decs->transAsTypedefField(e, base->trans(e), r);
}

void importdec::initialize(coenv &e, record *m, access *a)
{
  // Put the enclosing frame on the stack.
  if (!e.c.encode(m->getLevel()->getParent())) {
    em->error(getPos());
    *em << "import of struct '" << *m << "' is not in a valid scope";
  }
 
  // Encode the allocation. 
  e.c.encode(inst::makefunc,m->getInit());
  e.c.encode(inst::popcall);

  // Put the module into its memory location.
  a->encode(WRITE, getPos(), e.c);
  e.c.encode(inst::pop);
}



void importdec::prettyprint(ostream &out, int indent)
{
  prettyindent(out, indent);
  out << "importdec (" << *id << ")\n";
}

void importdec::loadFailed(coenv &)
{
  em->warning(getPos());
  *em << "could not load module of name '" << *id << "'";
  em->sync();
}

void importdec::trans(coenv &e)
{
  transAsField(e,0);
}

void importdec::transAsField(coenv &e, record *r)
{
  record *m = e.e.getModule(id);
  if (m == 0) {
    loadFailed(e);
    return;
  }

  // PRIVATE as only the body of a record, may refer to an imported record.
  access *a = r ? r->allocField(e.c.isStatic()) :
                  e.c.allocLocal();

  import *i = new import(m, a);

  // While the import is allocated as a field of the record, it is
  // only accessible inside the initializer of the record (and
  // nested functions and initializers), so there is no need to add it
  // to the environment maintained by the record.
  e.e.addImport(getPos(), id, i);

  // Add the initializer for the record.
  initialize(e, m, a);
}


void typedec::prettyprint(ostream &out, int indent)
{
  prettyname(out, "typedec",indent);

  body->prettyprint(out, indent+1);
}


void recorddec::prettyprint(ostream &out, int indent)
{
  prettyindent(out, indent);
  out << "structdec '" << *id << "'\n";

  body->prettyprint(out, indent+1);
}

function *recorddec::opType(record *r)
{
  function *ft = new function(primBoolean());
  ft->add(r);
  ft->add(r);

  return ft;
}

void recorddec::addOps(coenv &e, record *r)
{
  function *ft = opType(r);
  varEntry *ve=new varEntry(ft, new bltinAccess(run::boolMemEq));
  e.e.addVar(getPos(), symbol::trans("alias"), ve);
  e.e.addVar(getPos(), symbol::trans("=="), ve);
  e.e.addVar(getPos(), symbol::trans("!="),
      new varEntry(ft, new bltinAccess(run::boolMemNeq)));
}

void recorddec::trans(coenv &e)
{
  transAsField(e,0);
}  

void recorddec::transAsField(coenv &e, record *parent)
{
  record *r = parent ? parent->newRecord(id, e.c.isStatic()) :
                       e.c.newRecord(id);
                     
  if (parent)
    parent->addType(id, r);
  e.e.addType(getPos(), id, r);
  addOps(e,r);

  // Start translating the initializer.
  coder c=e.c.newRecordInit(r);
  coenv re(c,e.e);
  
  body->transAsRecordBody(re, r);
}  

  
} // namespace absyntax
