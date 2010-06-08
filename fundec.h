/*****
 * fundec.h
 * Andy Hammerlindl 2002/8/29
 *
 * Defines the semantics for defining functions.  Both the newexp syntax, and
 * the abbreviated C-style function definition.
 *****/

#ifndef FUNDEC_H
#define FUNDEC_H

#include "dec.h"
#include "exp.h"

namespace absyntax {

class formal : public absyn {
  ty *base;
  decidstart *start;
  bool Explicit;
  varinit *defval;

public:
  formal(position pos, ty *base, decidstart *start=0, varinit *defval=0,
         bool Explicit= false)
    : absyn(pos), base(base), start(start), Explicit(Explicit),
      defval(defval) {}

  virtual void prettyprint(ostream &out, Int indent);

  // Build the corresponding types::formal to put into a signature.
  types::formal trans(coenv &e, bool encodeDefVal, bool tacit=false);
  
  // Add the formal parameter to the environment to prepare for the
  // function body's translation.
  virtual void transAsVar(coenv &e, Int index);

  types::ty *getType(coenv &e, bool tacit=false);

  virtual void addOps(coenv &e, record *r);

  varinit *getDefaultValue() {
    return defval;
  }

  symbol getName() {
    return start ? start->getName() : symbol::nullsym;
  }

  bool getExplicit() {
    return Explicit;
  }
};

class formals : public absyn {
  //friend class funheader;

  mem::list<formal *> fields;
  formal *rest;

  void addToSignature(types::signature& sig,
                      coenv &e, bool encodeDefVal, bool tacit);
public:
  formals(position pos)
    : absyn(pos), rest(0) {}

  virtual ~formals() {}

  virtual void prettyprint(ostream &out, Int indent);

  virtual void add(formal *f) {
    fields.push_back(f);
  }

  virtual void addRest(formal *f) {
    rest = f;
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

  virtual void addOps(coenv &e, record *r);

  // Add the formal parameters to the environment to prepare for the
  // function body's translation.
  virtual void trans(coenv &e);
};

class fundef : public exp {
  ty *result;
  formals *params;
  stm *body;

  // If the fundef is part of a fundec, the name of the function is stored
  // here for debugging purposes.
  symbol id;

  friend class fundec;
  
public:
  fundef(position pos, ty *result, formals *params, stm *body)
    : exp(pos), result(result), params(params), body(body), id() {}

  virtual void prettyprint(ostream &out, Int indent);

  varinit *makeVarInit(types::function *ft);
  virtual void baseTrans(coenv &e, types::function *ft);
  virtual types::ty *trans(coenv &e);

  virtual types::function *transType(coenv &e, bool tacit);
  virtual types::function *transTypeAndAddOps(coenv &e, record *r, bool tacit);
  virtual types::ty *getType(coenv &e) {
    return transType(e, true);
  }
};

class fundec : public dec {
  symbol id;
  fundef fun;

public:
  fundec(position pos, ty *result, symbol id, formals *params, stm *body)
    : dec(pos), id(id), fun(pos, result, params, body)
  { fun.id = id; }

  void prettyprint(ostream &out, Int indent);

  void trans(coenv &e);

  void transAsField(coenv &e, record *r);
};
  
} // namespace absyntax

#endif
