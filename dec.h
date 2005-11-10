/*****
 * dec.h
 * Andy Hammerlindl 2002/8/29
 *
 * Represents the abstract syntax tree for declatations in the language.
 * Also included is abstract syntax for types as they are most often
 * used with declarations.
 *****/

#ifndef DEC_H
#define DEC_H

#include "symbol.h"
#include "absyn.h"
#include "name.h"
#include "varinit.h"
#include "modifier.h"

namespace trans {
class coenv;
class protoenv;
class varEntry;
class access;
}

namespace types {
class ty;
class formal;
class signature;
class function;
}

namespace vm {
class lambda;
}
namespace absyntax {

using mem::list;
using trans::coenv;
using trans::protoenv;
using trans::varEntry;
using trans::access;
using sym::symbol;

class ty : public absyn {
public:
  ty(position pos)
    : absyn(pos) {}

  virtual void prettyprint(ostream &out, int indent) = 0;

  // Returns the internal representation of the type.  This method can
  // be called by exp::getType which does not report errors, so tacit is
  // needed to silence errors in this case.
  virtual types::ty *trans(coenv &e, bool tacit = false) = 0;

  virtual trans::tyEntry *transAsTyEntry(coenv &e);
};

class nameTy : public ty {
  name *id;

public:
  nameTy(position pos, name *id)
    : ty(pos), id(id) {}

  nameTy(name *id)
    : ty(id->getPos()), id(id) {}

  void prettyprint(ostream &out, int indent);

  types::ty *trans(coenv &e, bool tacit = false);
  trans::tyEntry *transAsTyEntry(coenv &e);
};

class dimensions : public absyn {
  size_t depth;
public:
  dimensions(position pos)
    : absyn(pos), depth(1) {}

  void prettyprint(ostream &out, int indent);

  void increase()
    { depth++; }
  
  size_t size() {
    return depth;
  }

  types::ty *truetype(types::ty *base);
};

class arrayTy : public ty {
  ty *cell;
  dimensions *dims;

public:
  arrayTy(position pos, ty *cell, dimensions *dims)
    : ty(pos), cell(cell), dims(dims) {}

  arrayTy(name *id, dimensions *dims)
    : ty(dims->getPos()), cell(new nameTy(id)), dims(dims) {}

  void prettyprint(ostream &out, int indent);

  types::function *opType(types::ty* t);
  types::function *arrayType(types::ty* t);
  types::function *array2Type(types::ty* t);
  types::function *cellIntType(types::ty* t);
  types::function *sequenceType(types::ty* t, types::ty *ct);
  types::function *cellTypeType(types::ty* t);
  types::function *mapType(types::ty* t, types::ty *ct);
  void addOps(coenv &e, types::ty* t, types::ty *ct);
  
  types::ty *trans(coenv &e, bool tacit = false);
};

// Runnable is anything that can be executed by the program, including
// any declaration or statement.
class runnable : public absyn {
public:
  runnable(position pos)
    : absyn(pos) {}

  virtual void prettyprint(ostream &out, int indent) = 0;
  
  void markTrans(coenv &e)
  {
    markPos(e);
    trans(e);
  }
  
  /* Translates the stm or dec as if it were in a function definition. */
  virtual void trans(coenv &e) {
    transAsField(e, 0);
  }

  void markTransAsField(coenv &e, record *r)
  {
    markPos(e);
    transAsField(e,r);
  }

  /* Translate the runnable as in the lowest lexical scope of a record
   * definition.  If it is simply a statement, it will be added to the
   * record's initializer.  A declaration, however, will also have to
   * add a new type or field to the record.
   */
  virtual void transAsField(coenv &e, record *) = 0;

  virtual vm::lambda *transAsCodelet(coenv &e);

  // For functions that return a value, we must guarantee that they end
  // with a return statement.  This checks for that condition.
  virtual bool returns()
    { return false; }

  // Returns true if it is syntatically allowable to modify this
  // runnable by a PUBLIC or PRIVATE modifier. 
  virtual bool allowPermissions()
    { return false; }
}; 

class block : public runnable {
public:
  list<runnable *> stms;

  // If the runnables should be interpreted in their own scope.
  bool scope;

protected:
  void prettystms(ostream &out, int indent);

public:
  block(position pos, bool scope=true)
    : runnable(pos), scope(scope) {}

  // To ensure list deallocates properly.
  virtual ~block() {}

  void add(runnable *r) {
    stms.push_back(r);
  }

  void prettyprint(ostream &out, int indent);

  void trans(coenv &e);

  void transAsField(coenv &e, record *r);

  void transAsRecordBody(coenv &e, record *r);

  void transAsFile(coenv &e, record *r);

  // A block is guaranteed to return iff one of the runnables is guaranteed to
  // return.
  // This is conservative in that
  //
  // int f(int x)
  // {
  //   if (x==1) return 0;
  //   if (x!=1) return 1;
  // }
  //
  // is not guaranteed to return.
  bool returns();
};

class modifierList : public absyn {
  list<trans::permission> perms;
  list<trans::modifier> mods;

public:
  modifierList(position pos)
    : absyn(pos) {}

  virtual ~modifierList()
    {}

  void prettyprint(ostream &out, int indent);

  void add(trans::permission p)
  {
    perms.push_back(p);
  }

  void add(trans::modifier m)
  {
    mods.push_back(m);
  }

  /* True if a static or dynamic modifier is present.
   */
  bool staticSet();

  /* Says if the modifiers indicate static or dynamic. Prints error if
   * there are duplicates.
   */
  trans::modifier getModifier();

  /* Says if it is declared public, private, or read-only (default).
   * Prints error if there are duplicates.
   */
  trans::permission getPermission();
};

// Modifiers of static or dynamic can change the way declarations and
// statements are encoded.
class modifiedRunnable : public runnable {
  modifierList *mods;
  runnable *body;

public:
  modifiedRunnable(position pos, modifierList *mods, runnable *body)
    : runnable(pos), mods(mods), body(body)  {}

  modifiedRunnable(position pos, trans::permission perm, runnable *body)
    : runnable(pos), mods(new modifierList(pos)), body(body) {
    mods->add(perm);
  }

  void prettyprint(ostream &out, int indent);

  void transAsField(coenv &e, record *r);

  bool returns()
    { return body->returns(); }
};


class decidstart : public absyn {
protected:
  symbol *id;
  dimensions *dims;

public:
  decidstart(position pos, symbol *id, dimensions *dims = 0)
    : absyn(pos), id(id), dims(dims) {}

  virtual void prettyprint(ostream &out, int indent);

  virtual types::ty *getType(types::ty *base, coenv &, bool = false);
  virtual trans::tyEntry *getTyEntry(trans::tyEntry *base, coenv &e);

  virtual symbol *getName()
    { return id; }
};

// Forward declaration.
class formals;

class fundecidstart : public decidstart {
  formals *params;

public:
  fundecidstart(position pos,
                symbol *id,
		dimensions *dims = 0,
		formals *params = 0)
    : decidstart(pos, id, dims), params(params) {}

  void prettyprint(ostream &out, int indent);

  types::ty *getType(types::ty *base, coenv &e, bool tacit = false);
  virtual trans::tyEntry *getTyEntry(trans::tyEntry *base, coenv &e);
};

class decid : public absyn {
  decidstart *start;
  varinit *init;

  // Returns the default initializer for the type.
  access *defaultInit(coenv &e, types::ty *t);

public:
  decid(position pos, decidstart *start, varinit *init = 0)
    : absyn(pos), start(start), init(init) {}

  virtual void prettyprint(ostream &out, int indent);

  virtual void transAsField(coenv &e, record *r, types::ty *base);

  // Translate, but add the names in as types rather than variables. 
  virtual void transAsTypedefField(coenv &e, trans::tyEntry *base, record *r);
};

class decidlist : public absyn {
  list<decid *> decs;

public:
  decidlist(position pos)
    : absyn(pos) {}

  virtual ~decidlist() {}
  
  void add(decid *p) {
    decs.push_back(p);
  }

  virtual void prettyprint(ostream &out, int indent);

  virtual void transAsField(coenv &e, record *r, types::ty *base);

  // Translate, but add the names in as types rather than variables. 
  virtual void transAsTypedefField(coenv &e, trans::tyEntry *base, record *r);
};

class dec : public runnable {
public:
  dec(position pos)
    : runnable(pos) {}

  void prettyprint(ostream &out, int indent);

  // Declarations can be public or private.
  bool allowPermissions()
    { return true; }
};

void createVar(position pos, coenv &e, record *r,
               symbol *id, types::ty *t, varinit *init);

class vardec : public dec {
  ty *base;
  decidlist *decs;

public:
  vardec(position pos, ty *base, decidlist *decs)
    : dec(pos), base(base), decs(decs) {}

  void prettyprint(ostream &out, int indent);

  void transAsField(coenv &e, record *r)
  {
    decs->transAsField(e, r, base->trans(e));
  }

  // Translate, but add the names in as types rather than variables. 
  virtual void transAsTypedefField(coenv &e, record *r);
};

struct idpair : public absyn {
  symbol *src; // The name of the module to access.
  symbol *dest;  // What to call it in the local environment.

  idpair(position pos, symbol *id)
    : absyn(pos), src(id), dest(id) {}

  idpair(position pos, symbol *src, symbol *as, symbol *dest)
    : absyn(pos), src(src), dest(dest)
  {
    if (as!=symbol::trans("as")) {
      em->error(getPos());
      *em << "expected 'as'";
    }
  }

  void prettyprint(ostream &out, int indent);

  // Translates as: import src as dest;
  void transAsAccess(coenv &e, record *r);
  
  // Translates as: from _ unravel src as dest;
  // where _ is the qualifier record with source as its fields and types.
  void transAsUnravel(coenv &e, record *r,
                      protoenv &source, varEntry *qualifier);
};

struct idpairlist : public gc {
  mem::list<idpair *> base;

  void add(idpair *x) {
    base.push_back(x);
  }

  void prettyprint(ostream &out, int indent);

  void transAsAccess(coenv &e, record *r);

  void transAsUnravel(coenv &e, record *r,
                      protoenv &source, varEntry *qualifier);
};

extern idpairlist * const WILDCARD;


class accessdec : public dec {
  idpairlist *base;

public:
  accessdec(position pos, idpairlist *base)
    : dec(pos), base(base) {}

  void prettyprint(ostream &out, int indent);

  void transAsField(coenv &e, record *r) {
    base->transAsAccess(e,r);
  }
};

// Abstract base class for
//   from _ access _;  (importdec)
// and
//   from _ unravel _;  (unraveldec)
class fromdec : public dec {
protected:
  // Return the record from which the fields are taken.  If this returns 0, it
  // is assumed that an error occurred and was reported.
  virtual varEntry *getQualifier(coenv &e, record *r) = 0;
  idpairlist *fields;

public:
  fromdec(position pos, idpairlist *fields)
    : dec(pos), fields(fields) {}

  void prettyprint(ostream &out, int indent);

  void transAsField(coenv &e, record *r);
};

// An unravel declaration dumps fields and types of a record into the local
// scope.
class unraveldec : public fromdec {
  name *id;

  varEntry *getQualifier(coenv &e, record *);
public:
  unraveldec(position pos, name *id, idpairlist *fields)
    : fromdec(pos, fields), id(id) {}

  void prettyprint(ostream &out, int indent);
};

// An import declaration dumps fields and types of a module into the local
// scope.  It does not add the module as a variable in the local scope.
class importdec : public fromdec {
  symbol *id;

  varEntry *getQualifier(coenv &e, record *r);
public:
  importdec(position pos, symbol *id, idpairlist *fields)
    : fromdec(pos, fields), id(id) {}

  void prettyprint(ostream &out, int indent);
};

#if 0
class usedec : public dec {
  block base;

public:
  usedec(position pos, symbol *id, std::string filename)
    : dec(pos), base(pos, false) {
    base.add(new importdec(pos, id, filename));
    base.add(new explodedec(pos, new simpleName(pos, id)));
  }

  usedec(position pos, symbol *id)
    : dec(pos), base(pos, false) {
    base.add(new importdec(pos, id));
    base.add(new explodedec(pos, new simpleName(pos, id)));
  }

  void transAsField(coenv &e, record *r) {
    base.transAsField(e, r);
  }
};
#endif

// Parses the file given, and translates the resulting runnables as if they
// occurred at this place in the code.
class includedec : public dec {
  std::string filename;

public:
  includedec(position pos, std::string filename)
    : dec(pos), filename(filename) {}
  includedec(position pos, symbol *id)
    : dec(pos), filename(*id) {}

  void prettyprint(ostream &out, int indent);
  void loadFailed(coenv &e);

  void transAsField(coenv &e, record *r);
};

// Types defined from others in typedef.
class typedec : public dec {
  vardec *body;

public:
  typedec(position pos, vardec *body)
    : dec(pos), body(body) {}

  void prettyprint(ostream &out, int indent);

  void transAsField(coenv &e, record *r) {
    body->transAsTypedefField(e,r);
  }
};


// A struct declaration.
class recorddec : public dec {
  symbol *id;
  block *body;

  types::function *opType(record *r);
  void addOps(coenv &e, record *r);

public:
  recorddec(position pos, symbol *id, block *body)
    : dec(pos), id(id), body(body) {}

  virtual ~recorddec()
    {}

  void prettyprint(ostream &out, int indent);

  void transAsField(coenv &e, record *parent);
};

// Returns a runnable that facilitates the autoplain feature.
runnable *autoplainRunnable();

} // namespace absyntax

#endif

