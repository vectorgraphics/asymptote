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

  virtual void prettyprint(ostream &out, int indent);

  // Build the corresponding types::formal to put into a signature.
  types::formal trans(coenv &e, bool encodeDefVal, bool tacit=false);
  
  // Add the formal parameter to the environment to prepare for the
  // function body's translation.
  virtual void transAsVar(coenv &e, int index);

  types::ty *getType(coenv &e, bool tacit=false);

  varinit *getDefaultValue() {
    return defval;
  }

  // Report an error if there is a default value.
  // Used by newFunctionExp.
  bool reportDefault() {
    if (defval) {
      em->error(getPos());
      *em << "default value in anonymous function";
      return true;
    }
    else 
      return false;
  }

  symbol *getName() {
    return start ? start->getName() : 0;
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

  virtual void prettyprint(ostream &out, int indent);

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

  // Add the formal parameters to the environment to prepare for the
  // function body's translation.
  virtual void trans(coenv &e);

  // Report an error if there are default values.
  // Used by newFunctionExp.
  void reportDefaults();
};

class fundef : public exp {
  ty *result;
  formals *params;
  stm *body;
  
public:
  fundef(position pos, ty *result, formals *params, stm *body)
    : exp(pos), result(result), params(params), body(body) {}

  void prettyprint(ostream &out, int indent);

  types::ty *trans(coenv &e);

  types::function *transType(coenv &e, bool tacit);
  types::ty *getType(coenv &e) {
    return transType(e, true);
  }
};

class fundec : public dec {
  symbol *id;
  fundef fun;

public:
  fundec(position pos, ty *result, symbol *id, formals *params, stm *body)
    : dec(pos), id(id), fun(pos, result, params, body) {}

  void prettyprint(ostream &out, int indent);

  types::function *opType(types::function *f);
  void addOps(coenv &e, record *parent, types::function *f);
  
  void trans(coenv &e);

  void transAsField(coenv &e, record *r);
};
  
} // namespace absyntax

#endif
