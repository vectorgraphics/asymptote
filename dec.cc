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
#include "parser.h"

namespace absyntax {

using namespace trans;
using namespace types;


trans::tyEntry *ty::transAsTyEntry(coenv &e)
{
  return new trans::tyEntry(trans(e, false), 0);
}


void nameTy::prettyprint(ostream &out, int indent)
{
  prettyname(out, "nameTy",indent);

  id->prettyprint(out, indent+1);
}

types::ty *nameTy::trans(coenv &e, bool tacit)
{
  return id->typeTrans(e, tacit);
}

trans::tyEntry *nameTy::transAsTyEntry(coenv &e)
{
  return id->tyEntryTrans(e);
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
  
  e.e.addVar(symbol::trans("alias"),
      new varEntry(ft,new bltinAccess(run::arrayAlias)));

  if(dims->size() == 1) {
    e.e.addVar(symbol::trans("copy"),
	       new varEntry(ftarray,new bltinAccess(run::arrayCopy)));
    e.e.addVar(symbol::trans("concat"),
	       new varEntry(ftarray2,new bltinAccess(run::arrayConcat)));
    e.e.addVar(symbol::trans("sequence"),
	       new varEntry(ftsequence,new bltinAccess(run::arraySequence)));
    e.e.addVar(symbol::trans("map"),
	       new varEntry(ftmap,new bltinAccess(run::arrayFunction)));
  }
  if(dims->size() == 2) {
    e.e.addVar(symbol::trans("copy"),
	       new varEntry(ftarray,new bltinAccess(run::array2Copy)));
    e.e.addVar(symbol::trans("transpose"),
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

vm::lambda *runnable::transAsCodelet(coenv &e)
{
  coder c=e.c.newCodelet();
  coenv ce(c, e.e);
  trans(ce);
  return c.close();
}


void block::prettystms(ostream &out, int indent)
{
  for (list<runnable *>::iterator p = stms.begin(); p != stms.end(); ++p)
    (*p)->prettyprint(out, indent);
}

void block::prettyprint(ostream &out, int indent)
{
  prettyname(out,"block",indent);
  prettystms(out, indent+1);
}

void block::trans(coenv &e)
{
  if (scope) e.e.beginScope();
  for (list<runnable *>::iterator p = stms.begin(); p != stms.end(); ++p) {
    (*p)->markTrans(e);
  }
  if (scope) e.e.endScope();
}

void block::transAsField(coenv &e, record *r)
{
  if (scope) e.e.beginScope();
  for (list<runnable *>::iterator p = stms.begin(); p != stms.end(); ++p) {
    (*p)->markTransAsField(e, r);
  }
  if (scope) e.e.endScope();
}

void block::transAsRecordBody(coenv &e, record *r)
{
  transAsField(e, r);

  // Put record into finished state.
  e.c.encode(inst::pushclosure);
  e.c.close();
}

void block::transAsFile(coenv &e, record *r)
{
  if (settings::autoplain) {
    autoplainRunnable()->transAsField(e, r);
  }

  transAsRecordBody(e, r);
}
  
bool block::returns() {
  // Search for a returning runnable, starting at the end for efficiency.
  for (list<runnable *>::reverse_iterator p=stms.rbegin();
       p != stms.rend();
       ++p)
    if ((*p)->returns())
      return true;
  return false;
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
#if 0 // This is innocuous 
  if (p != READONLY && (!r || !body->allowPermissions())) {
    em->warning(pos);
    *em << "permission modifier is meaningless";
  }
#endif  
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

trans::tyEntry *decidstart::getTyEntry(trans::tyEntry *base, coenv &e)
{
  return dims ? new trans::tyEntry(getType(base->t,e,false), 0) :
                base;
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

trans::tyEntry *fundecidstart::getTyEntry(trans::tyEntry *base, coenv &e)
{
  return new trans::tyEntry(getType(base->t,e,false), 0);
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
    r->e.addVar(id, ent);
  e.e.addVar(id, ent);
  
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
    r->e.addVar(id, ent);
  e.e.addVar(id, ent);
  
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

void decid::transAsTypedef(coenv &e, trans::tyEntry *base)
{
  transAsTypedefField(e, base, 0);
}

void decid::transAsTypedefField(coenv &e, trans::tyEntry *base, record *r)
{
  trans::tyEntry *ent = start->getTyEntry(base, e);
  assert(ent && ent->t);

  if (init) {
    em->error(getPos());
    *em << "type definition cannot have initializer";
  }
   
  // Add to type to record and environment.
  if (r)
    r->e.addType(start->getName(), ent);
  e.e.addType(start->getName(), ent);
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

void decidlist::transAsTypedef(coenv &e, trans::tyEntry *base)
{
  for (list<decid *>::iterator p = decs.begin(); p != decs.end(); ++p)
    (*p)->transAsTypedef(e, base);
}

void decidlist::transAsTypedefField(coenv &e, trans::tyEntry *base, record *r)
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
  decs->transAsTypedef(e, base->transAsTyEntry(e));
}

void vardec::transAsTypedefField(coenv &e, record *r)
{
  decs->transAsTypedefField(e, base->transAsTyEntry(e), r);
}

// Helper class for imports.  This essentially evaluates to the run::loadModule
// function.  However, that function returns different types of records
// depending on the filename given to it, so we cannot add it to the
// environment.  Instead, we explicitly tell it what types::record it is
// returning for each use.
class loadModuleExp : public exp {
  record *imp;
  function *ft;

public:
  loadModuleExp(position pos, record *imp)
    : exp(pos), imp(imp), ft(new function(imp))
  {
    ft->add(primString());
  }

  types::ty *trans(coenv &e) {
    em->compiler(getPos());
    *em << "trans called for loadModuleExp";
    return primError();
  }

  void transCall(coenv &e, types::ty *t) {
    assert(equivalent(t, ft));
    e.c.encode(inst::builtin, run::loadModule);
  }

  types::ty *getType(coenv &e) {
    return ft;
  }

  exp *evaluate(coenv &, types::ty *) {
    // Don't alias.
    return this;
  }
};

void importdec::prettyprint(ostream &out, int indent)
{
  prettyindent(out, indent);
  out << "importdec (" << "'" << filename << "' as " << *id << ")\n";
}

void importdec::trans(coenv &e)
{
  transAsField(e,0);
}

void importdec::loadFailed(coenv &)
{
  em->warning(getPos());
  *em << "could not load module of name '" << *id << "'";
  em->sync();
}

void importdec::transAsField(coenv &e, record *r)
{
  record *imp=e.e.getModule(id, filename);
  if (!imp) {
    loadFailed(e);
  }
  else {
    // Create a varinit that evaluates to the module.
    // This is effectively the expression "loadModule(filename)".
    callExp init(getPos(), new loadModuleExp(getPos(), imp),
                           new stringExp(getPos(), filename));

    // Add the variable to the environment.
    // This is effectively a variable declaration of the form
    //
    // imp id=loadModule(filename);
    //
    // except that the type "imp" of the module is not in the type
    // environment.
    addVar(pos, e, r, id, imp, &init);
  }
}


void explodedec::prettyprint(ostream &out, int indent)
{
  prettyname(out, "explodedec", indent);
  id->prettyprint(out, indent+1);
}

void explodedec::trans(coenv &e)
{
  transAsField(e,0);
}

void explodedec::transAsField(coenv &e, record *r)
{
  record *qualifier=dynamic_cast<record *>(id->getType(e, false));
  if (!qualifier) {
    em->error(getPos());
    *em << "'" << *(id->getName()) << "' is not a record";
  }
  else {
    varEntry *v=id->getVarEntry(e);
    if (r)
      r->e.add(qualifier->e, v, e.c);
    e.e.add(qualifier->e, v, e.c);
  }
}


void includedec::prettyprint(ostream &out, int indent)
{
  prettyindent(out, indent);
  out << "includedec ('" << filename << "')\n";
}

void includedec::trans(coenv &e)
{
  transAsField(e,0);
}

void includedec::loadFailed(coenv &)
{
  em->warning(getPos());
  *em << "could not parse file of name '" << filename << "'";
  em->sync();
}

void includedec::transAsField(coenv &e, record *r)
{
  file *ast = parser::parseFile(filename);
  em->sync();

  // The runnables will be run, one at a time, without any additional scoping.
  ast->transAsField(e, r);
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
  e.e.addVar(symbol::trans("alias"), ve);
  e.e.addVar(symbol::trans("=="), ve);
  e.e.addVar(symbol::trans("!="),
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
                     
  tyEntry *ent = new trans::tyEntry(r,0);

  if (parent)
    parent->e.addType(id, ent);
  e.e.addType(id, ent);
  addOps(e,r);

  // Start translating the initializer.
  coder c=e.c.newRecordInit(r);
  coenv re(c,e.e);
  
  body->transAsRecordBody(re, r);
}  

runnable *autoplainRunnable() {
  // Private import plain;
  static usedec ap(position(), symbol::trans("plain"));
  static modifiedRunnable mr(position(), trans::PRIVATE, &ap);

  return &mr;
}

} // namespace absyntax
