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
#include "exp.h"

namespace trans {
class coenv;
class access;
}

namespace types
{
class ty;
class signature;
class function;
}

namespace as {

using trans::coenv;
using trans::access;
using sym::symbol;

class ty : public absyn {
public:
  ty(position pos)
    : absyn(pos) {}

  virtual void prettyprint(ostream &out, int indent) = 0;

  // Returns the internal representation of the type.  This method can
  // be called by stm::getType which does not report errors, so tacit is
  // needed to silence errors in this case.
  virtual types::ty *trans(coenv &e, bool tacit = false) = 0;

  // Finds the import that the type is imported from.
  // This necessary for record allocations.
  // Returns 0 if the type ultimately refers to no imports.
  virtual trans::import *getImport(coenv &e) = 0;
};

class nameTy : public ty {
  name *id;

public:
  nameTy(position pos, name *id)
    : ty(pos), id(id) {}

  void prettyprint(ostream &out, int indent);

  types::ty *trans(coenv &e, bool tacit = false);

  trans::import *getImport(coenv &e)
  {
    return id->typeGetImport(e);
  }
};

class arrayTy : public ty {
  ty *cell;
  dimensions *dims;

public:
  arrayTy(position pos, ty *cell, dimensions *dims)
    : ty(pos), cell(cell), dims(dims) {}

  void prettyprint(ostream &out, int indent);

  types::function *opType(types::ty* t);
  types::function *arrayType(types::ty* t);
  types::function *cellIntType(types::ty* t);
  types::function *sequenceType(types::ty* t, types::ty *ct);
  types::function *cellTypeType(types::ty* t);
  types::function *evalType(types::ty* t, types::ty *ct);
  void addOps(coenv &e, types::ty* t, types::ty *ct);
  
  types::ty *trans(coenv &e, bool tacit = false);

  trans::import *getImport(coenv &e)
  {
    return cell->getImport(e);
  }
};

// Runnable is anything that can be executed by the program, including
// any declaration or statement.
class runnable : public absyn {
public:
  runnable(position pos)
    : absyn(pos) {}

  virtual void prettyprint(ostream &out, int indent) = 0;
  
  virtual void markTrans(coenv &e)
  {
    markPos(e);
    trans(e);
  }
  
  /* Translates the stm or dec as if it were in a function definition. */
  virtual void trans(coenv &e) = 0;

  virtual void markTransAsField(coenv &e, record *r)
  {
    markPos(e);
    transAsField(e,r);
  }

  /* Translate the runnable as a in the lowest lexical scope of a record 
   * definition.  If it is simply a statement, it will be added to the
   * record's initializer.  A declaration, however, will also have to
   * add a new type or field to the record.
   */
  virtual void transAsField(coenv &e, record *) {
    // By default, translate as normal.
    trans(e);
  }

  // For functions that return a value, we must guarantee that they end
  // with a return statement.  This checks for that condition.
  virtual bool returns()
    { return false; }

  // Returns true if it is syntatically allowable to modify this
  // runnable by a PUBLIC or PRIVATE modifier. 
  virtual bool allowPermissions()
    { return false; }
}; 

class modifierList : public absyn {
  std::list<int> mods;

public:
  modifierList(position pos)
    : absyn(pos) {}

  virtual ~modifierList()
    {}

  void prettyprint(ostream &out, int indent);

  void add(int m)
  {
    mods.push_back(m);
  }

  /* True if a static or dynamic modifier is present.
   */
  bool staticSet();

  /* Says if the modifiers indicate static or dynamic. Prints error if
   * there are duplicates.
   */
  bool isStatic();

  /* Says if it is declared public, private, or read-only (default).
   * Prints error if there are duplicates.
   */
  trans::permission getPermission();
};

// Mpdifiers of static or dynamic can change the way declarations and
// statements are encoded.
class modifiedRunnable : public runnable {
  modifierList *mods;
  runnable *body;

public:
  modifiedRunnable(position pos, modifierList *mods, runnable *body)
    : runnable(pos), mods(mods), body(body)  {}

  void prettyprint(ostream &out, int indent);

  void trans(coenv &e);
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

  virtual void trans(coenv &e, types::ty *base);

  virtual void transAsField(coenv &e, record *r, types::ty *base);

  // Translate, but add the names in as types rather than variables. 
  virtual void transAsTypedef(coenv &e, types::ty *base);
  virtual void transAsTypedefField(coenv &e, types::ty *base, record *r);
};

class decidlist : public absyn {
  std::list<decid *> decs;

public:
  decidlist(position pos)
    : absyn(pos) {}

  virtual ~decidlist() {}
  
  void add(decid *p) {
    decs.push_back(p);
  }

  virtual void prettyprint(ostream &out, int indent);

  virtual void trans(coenv &e, types::ty *base);

  virtual void transAsField(coenv &e, record *r, types::ty *base);

  // Translate, but add the names in as types rather than variables. 
  virtual void transAsTypedef(coenv &e, types::ty *base);
  virtual void transAsTypedefField(coenv &e, types::ty *base, record *r);
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

class vardec : public dec {
  ty *base;
  decidlist *decs;

public:
  vardec(position pos, ty *base, decidlist *decs)
    : dec(pos), base(base), decs(decs) {}

  void prettyprint(ostream &out, int indent);

  void trans(coenv &e)
  {
    decs->trans(e, base->trans(e));
  }

  void transAsField(coenv &e, record *r)
  {
    decs->transAsField(e, r, base->trans(e));
  }

  // Translate, but add the names in as types rather than variables. 
  virtual void transAsTypedef(coenv &e);
  virtual void transAsTypedefField(coenv &e, record *r);
};

class importdec : public dec {
  symbol *id;

  void initialize(coenv &e, record *m, access *a);

public:
  importdec(position pos, symbol *id)
    : dec(pos), id(id) {}

  void prettyprint(ostream &out, int indent);
  void loadFailed(coenv &e);

  void trans(coenv &e);

  void transAsField(coenv &e, record *r);

  // PUBLIC and PRIVATE modifiers are meaningless to imports, so we do
  // not allow them.
  bool allowPermissions()
    { return false; }
};

// Types defined from others in typedef.
class typedec : public dec {
  vardec *body;

public:
  typedec(position pos, vardec *body)
    : dec(pos), body(body) {}

  void prettyprint(ostream &out, int indent);

  void trans(coenv &e) {
    body->transAsTypedef(e);
  }
  void transAsField(coenv &e, record *r) {
    body->transAsTypedefField(e,r);
  }
};


class formal : public absyn {
  ty *base;
  decidstart *start;
  // NOTE: expressions used in default values are translated into vm
  // code at the call location, not the function definition location.
  // This should be changed, using codelets or small helper functions.
  // Tom: Or not. The most common use of default arguments is to pass
  // a standard value like currentpicture to a function. The current
  // implementation allows one to do something like
  // {
  //   picture currentpicture;
  //   // Drawing Code
  // }
  // and have all the drawing calls draw on the just declared picture
  // rather than the global currentpicture.
  varinit *defval;

public:
  formal(position pos, ty *base, decidstart *start=0, varinit *defval=0)
    : absyn(pos), base(base), start(start), defval(defval) {}

  virtual void prettyprint(ostream &out, int indent);

  virtual types::ty *getType(coenv &e, bool tacit = false);

  virtual varinit *getDefaultValue() {
    return defval;
  }

  virtual symbol *getName() {
    return start ? start->getName() : 0;
  } 
};

class formals : public absyn {
  //friend class funheader;

  std::list<formal *> fields;

public:
  formals(position pos)
    : absyn(pos) {}

  virtual ~formals() {}

  virtual void prettyprint(ostream &out, int indent);

  virtual void add(formal *f) {
    fields.push_back(f);
  }

  // Returns the types of each parameter as a signature.
  // encodeDefVal means that it will also encode information regarding
  // the default values into the signature
  types::signature *getSignature(coenv &e,
                                 bool encodeDefVal = false,
                                 bool tacit = false);

  // Returns the corresponding function type, assuming it has a return
  // value of "result."
  types::function *getType(types::ty *result, coenv &e,
                           bool encodeDefVal = false,
                           bool tacit = false);

  // Add the formal parameters to the environment to prepare for the
  // function body's translation.
  virtual void trans(coenv &e);

  // Report an error if there are default values.
  // Used by newFunctionExp.
  void reportDefaults();
};

class fundec : public dec {
  ty *result;
  symbol *id;
  formals *params;
  stm *body;

public:
  fundec(position pos, ty *result, symbol *id, formals *params, stm *body)
    : dec(pos), result(result), id(id), params(params), body(body) {}

  void prettyprint(ostream &out, int indent);

  types::function *opType(types::function *f);
  void addOps(coenv &e, types::function *f);
  
  void trans(coenv &e);

  void transAsField(coenv &e, record *r);
};
  

// A struct declaration.
class recorddec : public dec {
  symbol *id;
  blockStm *body;

  types::function *opType(record *r);
  void addOps(coenv &e, record *r);

public:
  recorddec(position pos, symbol *id, blockStm *body)
    : dec(pos), id(id), body(body) {}

  virtual ~recorddec()
    {}

  void prettyprint(ostream &out, int indent);

  void trans(coenv &e);

  void transAsField(coenv &e, record *parent);
};


} // namespace as

#endif

