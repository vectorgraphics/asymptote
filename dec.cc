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
  return new function(primBoolean(),types::formal(t,"a"),types::formal(t,"b"));
}

function *arrayTy::arrayType(types::ty* t)
{
  return new function(t,types::formal(t,"a"));
}

function *arrayTy::array2Type(types::ty* t)
{
  return new function(t,types::formal(t,"a"),types::formal(t,"b"));
}

function *arrayTy::cellIntType(types::ty* t)
{
  return new function(t,primInt());
}
  
function *arrayTy::sequenceType(types::ty* t, types::ty *ct)
{
  return new function(t,types::formal(cellIntType(ct),"f"),
		      types::formal(primInt(),"n"));
}

function *arrayTy::cellTypeType(types::ty* t)
{
  return new function(t,t);
}
  
function *arrayTy::mapType(types::ty* t, types::ty *ct)
{
  return new function(t,types::formal(cellTypeType(ct),"f"),
		      types::formal(t,"a"));
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
  markTrans(ce);
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
  e.c.closeRecord();
}

void block::transAsFile(coenv &e, record *r)
{
  if (settings::getSetting<bool>("autoplain")) {
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

  body->transAsField(e,r);

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


varEntry *makeVarEntry(coenv &e, record *r, types::ty *t)
{
  access *a = r ? r->allocField(e.c.isStatic()) :
                  e.c.allocLocal();

  return r ? new varEntry(t, a, e.c.getPermission(), r) :
             new varEntry(t, a);
}

void addVar(coenv &e, record *r, varEntry *v, symbol *id)
{
  // Add to the record so it can be accessed when qualified; add to the
  // environment so it can be accessed unqualified in the scope of the
  // record definition.
  if (r)
    r->e.addVar(id, v);
  e.e.addVar(id, v);
}

void initializeVar(position pos, coenv &e, record *,
                   varEntry *v, types::ty *t, varinit *init)
{
  if (init)
    init->transToType(e, t);
  else {
    definit d(pos);
    d.transToType(e, t);
  }
  
  v->getLocation()->encode(WRITE, pos, e.c);
  e.c.encode(inst::pop);
}

void createVar(position pos, coenv &e, record *r,
               symbol *id, types::ty *t, varinit *init)
{
  varEntry *v=makeVarEntry(e, r, t);
  addVar(e, r, v, id);
  initializeVar(pos, e, r, v, t, init);
}

void createVarOutOfOrder(position pos, coenv &e, record *r,
                         symbol *id, types::ty *t, varinit *init)
{
  varEntry *v=makeVarEntry(e, r, t);
  initializeVar(pos, e, r, v, t, init);
  addVar(e, r, v, id);
}


void decid::transAsField(coenv &e, record *r, types::ty *base)
{
  types::ty *t = start->getType(base, e);
  assert(t);
  if (t->kind == ty_void) {
    em->compiler(getPos());
    *em << "can't declare variable of type void";
  }

  createVarOutOfOrder(getPos(), e, r, start->getName(), t, init);
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

void decidlist::transAsField(coenv &e, record *r, types::ty *base)
{
  for (list<decid *>::iterator p = decs.begin(); p != decs.end(); ++p)
    (*p)->transAsField(e, r, base);
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
    : exp(pos), imp(imp), ft(new function(imp,primString())) {}

  types::ty *trans(coenv &) {
    em->compiler(getPos());
    *em << "trans called for loadModuleExp";
    return primError();
  }

  void transCall(coenv &e, types::ty *t) {
    assert(equivalent(t, ft));
    e.c.encode(inst::builtin, run::loadModule);
  }

  types::ty *getType(coenv &) {
    return ft;
  }

  exp *evaluate(coenv &, types::ty *) {
    // Don't alias.
    return this;
  }
};

// Creates a local variable to hold the import and translate the accessing of
// the import, but doesn't add the import to the environment.
varEntry *accessModule(position pos, coenv &e, record *r, symbol *id)
{
  record *imp=e.e.getModule(id, (mem::string)*id);
  if (!imp) {
    em->error(pos);
    *em << "could not load module of name '" << *id << "'";
    em->sync();
    return 0;
  }
  else {
    // Create a varinit that evaluates to the module.
    // This is effectively the expression "loadModule(filename)".
    callExp init(pos, new loadModuleExp(pos, imp),
                      new stringExp(pos, *id));

    varEntry *v=makeVarEntry(e, r, imp);
    initializeVar(pos, e, r, v, imp, &init);
    return v;
  }
}


void idpair::prettyprint(ostream &out, int indent)
{
  prettyindent(out, indent);
  out << "idpair (" << "'" << *src << "' as " << *dest << ")\n";
}

void idpair::transAsAccess(coenv &e, record *r)
{
  checkValidity();

  varEntry *v=accessModule(getPos(), e, r, src);
  if (v)
    addVar(e, r, v, dest);
}

void idpair::transAsUnravel(coenv &e, record *r,
                            protoenv &source, varEntry *qualifier)
{
  checkValidity();

  if (r)
    r->e.add(src, dest, source, qualifier, e.c);
  if (!e.e.add(src, dest, source, qualifier, e.c)) {
    em->error(getPos());
    *em << "no matching types or fields of name '" << *src << "'";
  }
}


void idpairlist::prettyprint(ostream &out, int indent)
{
  for (mem::list<idpair *>::iterator p=base.begin();
       p != base.end();
       ++p)
    (*p)->prettyprint(out, indent);
}

void idpairlist::transAsAccess(coenv &e, record *r)
{
  for (mem::list<idpair *>::iterator p=base.begin();
       p != base.end();
       ++p)
    (*p)->transAsAccess(e,r);
}

void idpairlist::transAsUnravel(coenv &e, record *r,
                                protoenv &source, varEntry *qualifier)
{
  for (mem::list<idpair *>::iterator p=base.begin();
       p != base.end();
       ++p)
    (*p)->transAsUnravel(e,r,source,qualifier);
}

idpairlist * const WILDCARD = 0;

void accessdec::prettyprint(ostream &out, int indent)
{
  prettyname(out, "accessdec", indent);
  base->prettyprint(out, indent+1);
}


void fromdec::prettyprint(ostream &out, int indent)
{
  prettyname(out, "fromdec", indent);
  fields->prettyprint(out, indent+1);
}

void fromdec::transAsField(coenv &e, record *r)
{
  varEntry *v=getQualifier(e,r);
  if (v) {
    record *qt=dynamic_cast<record *>(v->getType());
    if (!qt) {
      em->error(getPos());
      *em << "qualifier is not a record";
    }
    else {
      if (fields==WILDCARD) {
        if (r)
          r->e.add(qt->e, v, e.c);
        e.e.add(qt->e, v, e.c);
      }
      else
        fields->transAsUnravel(e, r, qt->e, v);
    }
  }
}


void unraveldec::prettyprint(ostream &out, int indent)
{
  prettyname(out, "unraveldec", indent);
  id->prettyprint(out, indent+1);
  idpairlist *f=this->fields;
  if(f) f->prettyprint(out, indent+1);
}

varEntry *unraveldec::getQualifier(coenv &e, record *)
{
  // To report errors.
  id->getType(e, false);

  return id->getVarEntry(e);
}

void fromaccessdec::prettyprint(ostream &out, int indent)
{
  prettyindent(out, indent);
  out << "fromaccessdec '" << *id << "'\n";
  idpairlist *f=this->fields;
  if(f) f->prettyprint(out, indent+1);
}

varEntry *fromaccessdec::getQualifier(coenv &e, record *r)
{
  return accessModule(getPos(), e, r, id);
}

void importdec::prettyprint(ostream &out, int indent)
{
  prettyname(out, "importdec", indent);
  base.prettyprint(out, indent+1);
}

void includedec::prettyprint(ostream &out, int indent)
{
  prettyindent(out, indent);
  out << "includedec ('" << filename << "')\n";
}

void includedec::loadFailed(coenv &)
{
  em->warning(getPos());
  *em << "could not parse file of name '" << filename << "'";
  em->sync();
}

void includedec::transAsField(coenv &e, record *r)
{
  if(settings::verbose > 1)
    std::cerr << "Including " <<  filename << std::endl;
  
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
  return new function(primBoolean(),types::formal(r,"a"),
		      types::formal(r,"b"));
}

void recorddec::addOps(coenv &e, record *parent, record *r)
{
  function *ft = opType(r);
  varEntry *ve=new varEntry(ft, new bltinAccess(run::boolMemEq));
  addVar(e,parent,ve,symbol::trans("alias"));
  addVar(e,parent,ve,symbol::trans("=="));
  addVar(e,parent,new varEntry(ft, new bltinAccess(run::boolMemNeq)),
	 symbol::trans("!="));
}

void recorddec::transAsField(coenv &e, record *parent)
{
  record *r = parent ? parent->newRecord(id, e.c.isStatic()) :
                       e.c.newRecord(id);
                     
  tyEntry *ent = new trans::tyEntry(r,0);

  if (parent)
    parent->e.addType(id, ent);
  e.e.addType(id, ent);
  addOps(e,parent,r);

  // Start translating the initializer.
  coder c=e.c.newRecordInit(r);
  coenv re(c,e.e);
  
  body->transAsRecordBody(re, r);
}  

runnable *autoplainRunnable() {
  // Private import plain;
  position pos=position();
  static importdec ap(pos, new idpair(pos, symbol::trans("plain")));
  static modifiedRunnable mr(pos, trans::PRIVATE, &ap);

  return &mr;
}

} // namespace absyntax
