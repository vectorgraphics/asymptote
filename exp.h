/*****
 * exp.h
 * Andy Hammerlindl 2002/8/19
 *
 * Represents the abstract syntax tree for the expressions in the
 * language.  this is translated into virtual machine code using trans()
 * and with the aid of the environment class.
 *****/

#ifndef EXP_H
#define EXP_H

#include "types.h"
#include "symbol.h"
#include "absyn.h"
#include "varinit.h"
#include "name.h"
#include "guideflags.h"

namespace trans {
class coenv;
class application;
}

namespace absyntax {

using mem::list;
using trans::coenv;
using trans::application;
using trans::access;
using sym::symbol;
using types::record;
using types::array;

class exp : public varinit {
protected:
  // The cached type (from a call to cgetType).
  types::ty *ct;
public:
  exp(position pos)
    : varinit(pos), ct(0) {}

  void prettyprint(ostream &out, int indent);

  // When reporting errors with function calls, it is nice to say "no
  // function f(int)" instead of "no function matching signature
  // (int)."  Hence, this method returns the name of the expression if
  // there is one.
  virtual symbol *getName()
  {
    return 0;
  }

  // Checks if expression can be used as the right side of a scale
  // expression.  ie. 3sin(x)
  virtual bool scalable() { return true; }
  
  // Translates the expression to the given target type.  This should only be
  // called with a type returned by getType().  It does not perform implicit
  // casting.
  virtual void transAsType(coenv &e, types::ty *target);

  // Translates the expression to the given target type, possibly using an
  // implicit cast.
  void transToType(coenv &e, types::ty *target);

  // Translates the expression and returns the resultant type.
  // For some expressions, this will be ambiguous and return an error.
  // Trans may only return ty_error, if it (or one of its recursively
  // called children in the syntax tree) reported an error to em.
  virtual types::ty *trans(coenv &) = 0;

  // getType() figures out the type of the expression without translating
  // the code into the virtual machine language or reporting errors to em.
  // This must follow a few rules to ensure proper translation:
  //   1. If this returns a valid type, t, trans(e) must return t or
  //      report an error, and transToType(e, t) must run either reporting
  //      an error or reporting no error and yielding the same result as
  //      trans(e).
  //   2. If this returns a superposition of types (ie. for overloaded
  //      functions), trans must not return a singular type, and every
  //      type in the superposition must run without error properly
  //      if fed to transAsType(e, t).
  //   3. If this returns ty_error, then so must a call to trans(e) and any
  //      call to trans, transAsType, or transToType must report an error
  //      to em.
  //   4. Any call to transAsType(e, t) with a type that is not returned by
  //      getType() (or one of the subtypes in case of a superposition)
  //      must report an error.
  //      Any call to transToType(e, t) with a type that is not returned by
  //      getType() (or one of the subtypes in case of a superposition)
  //      or any type not implicitly castable from the above must report an
  //      error.
  virtual types::ty *getType(coenv &) = 0;

  // Same result as getType, but caches the result so that subsequent
  // calls are faster.  For this to work correctly, the expression should
  // only be used in one place, so the environment doesn't change between
  // calls.
  virtual types::ty *cgetType(coenv &e) {
    return ct ? ct : ct = getType(e);
  }

  // The expression is being used as an address to write to.  This writes code
  // so that the value on top of stack is put into the address (but not popped
  // off the stack).
  virtual void transWrite(coenv &, types::ty *) {
    em->error(getPos());
    *em << "expression cannot be used as an address";
  }

  // Translates code for calling a function.  The arguments, in the order they
  // appear in the function's signature, must all be on the stack.
  virtual void transCall(coenv &e, types::ty *target);

  // This is used to ensure the proper order and number of evaluations.  When
  // called, it immediately translates code to perform the side-effects
  // consistent with a corresponding call to transAsType(e, target).
  //
  // The return value, called an evaluation for lack of a better name, is
  // another expression that responds to the trans methods exactly as would the
  // original expression, but without producing side-effects.  It is also no
  // longer overloaded, due to the resolution effected by giving a target type
  // to evaluate().
  //
  // The methods transAsType, transWrite, and transCall of the evaluation must
  // be called with the same target type as the original call to evaluate.
  // When evaluate() is called during the translation of a function, that
  // function must still be in translation when the evaluation is translated.
  // 
  // The base implementation uses a tempExp (see below).  This is
  // sufficient for most expressions.
  virtual exp *evaluate(coenv &e, types::ty *target);

  // NOTE: could add a "side-effects" method which says if the expression has
  // side-effects.  This might allow some small optimizations in translating.
};

class tempExp : public exp {
  access *a;
  types::ty *t;

public:
  tempExp(coenv &e, varinit *v, types::ty *t);

  types::ty *trans(coenv &e);

  types::ty *getType(coenv &) {
    return t;
  }
};

// Wrap a varEntry so that it can be used as an expression.
class varEntryExp : public exp {
  trans::varEntry *v;
public:
  varEntryExp(position pos, trans::varEntry *v) 
    : exp(pos), v(v) {}
  varEntryExp(position pos, types::ty *t, access *a);
  varEntryExp(position pos, types::ty *t, vm::bltin f);

  types::ty *getType(coenv &);
  types::ty *trans(coenv &e);
  
  void transAct(action act, coenv &e, types::ty *target);
  void transAsType(coenv &e, types::ty *target);
  void transWrite(coenv &e, types::ty *target);
  void transCall(coenv &e, types::ty *target);
};

class nameExp : public exp {
  name *value;

public:
  nameExp(position pos, name *value)
    : exp(pos), value(value) {}

  nameExp(position pos, symbol *id)
    : exp(pos), value(new simpleName(pos, id)) {}

  void prettyprint(ostream &out, int indent);

  symbol *getName()
  {
    return value->getName();
  }

  void transAsType(coenv &e, types::ty *target) {
    value->varTrans(trans::READ, e, target);
    
    // After translation, the cached type is no longer needed and should be
    // garbage collected.  This could presumably be done in every class derived
    // from exp, but here it is most important as nameExp can have heavily
    // overloaded types cached.
    ct=0;
  }

  types::ty *trans(coenv &e) {
    types::ty *t=cgetType(e);
    if (t->kind == types::ty_error) {
      em->error(getPos());
      *em << "no matching variable \'" << *value << "\'";
      return types::primError();
    }
    if (t->kind == types::ty_overloaded) {
      em->error(getPos());
      *em << "use of variable \'" << *value << "\' is ambiguous";
      return types::primError();
    }
    else {
      transAsType(e, t);
      return t;
    }
  }

  types::ty *getType(coenv &e) {
    types::ty *t=value->varGetType(e);
    return t ? t : types::primError();
  }

  void transWrite(coenv &e, types::ty *target) {
    value->varTrans(trans::WRITE, e, target);

    ct=0;  // See note in transAsType.
  }
  
  void transCall(coenv &e, types::ty *target) {
    value->varTrans(trans::CALL, e, target);

    ct=0;  // See note in transAsType.
  }

  exp *evaluate(coenv &, types::ty *) {
    // Names have no side-effects.
    return this;
  }
};

// Most fields accessed are handled as parts of qualified names, but in cases
// like f().x or (new t).x, a separate expression is needed.
class fieldExp : public nameExp {
  exp *object;
  symbol *field;

  // fieldExp has a lot of common functionality with qualifiedName, so we
  // essentially hack qualifiedName, by making our object expression look
  // like a name.
  class pseudoName : public name {
    exp *object;

  public:
    pseudoName(exp *object)
      : name(object->getPos()), object(object) {}

    // As a variable:
    void varTrans(trans::action act, coenv &e, types::ty *target) {
      assert(act == trans::READ);
      object->transToType(e, target);
    }
    types::ty *varGetType(coenv &e) {
      return object->getType(e);
    }

    // As a type:
    types::ty *typeTrans(coenv &, bool tacit = false) {
      if (!tacit) {
        em->error(getPos());
        *em << "expression is not a type";
      }
      return types::primError();
    }

    trans::varEntry *getVarEntry(coenv &) {
      em->compiler(getPos());
      *em << "expression cannot be used as part of a type";
      return 0;
    }

    trans::tyEntry *tyEntryTrans(coenv &) {
      em->compiler(getPos());
      *em << "expression cannot be used as part of a type";
      return 0;
    }

    trans::frame *tyFrameTrans(coenv &) {
      return 0;
    }

    void prettyprint(ostream &out, int indent);
    void print(ostream& out) const {
      out << "<exp>";
    }
    symbol *getName() {
      return object->getName();
    }
  };

  // Try to get this into qualifiedName somehow.
  types::ty *getObject(coenv &e);

public:
  fieldExp(position pos, exp *object, symbol *field)
    : nameExp(pos, new qualifiedName(pos,
                                     new pseudoName(object),
                                     field)),
      object(object), field(field) {}

  void prettyprint(ostream &out, int indent);

  symbol *getName()
  {
    return field;
  }

  exp *evaluate(coenv &e, types::ty *) {
    // Evaluate the object.
    return new fieldExp(getPos(),
                        new tempExp(e, object, getObject(e)),
                        field);
  }
};

class subscriptExp : public exp {
  exp *set;
  exp *index;

  array *getArrayType(coenv &e);
  array *transArray(coenv &e);

public:
  subscriptExp(position pos, exp *set, exp *index)
    : exp(pos), set(set), index(index) {}

  void prettyprint(ostream &out, int indent);

  types::ty *trans(coenv &e);
  types::ty *getType(coenv &e);
  void transWrite(coenv &e, types::ty *target);

  exp *evaluate(coenv &e, types::ty *) {
    return new subscriptExp(getPos(),
                            new tempExp(e, set, getArrayType(e)),
                            new tempExp(e, index, types::primInt()));
  }
};

// The expression "this," that evaluates to the lexically enclosing record.
class thisExp : public exp {
public:
  thisExp(position pos)
    : exp(pos) {}

  void prettyprint(ostream &out, int indent);

  types::ty *trans(coenv &e);
  types::ty *getType(coenv &e);
};

class literalExp : public exp {
public:
  literalExp(position pos)
    : exp(pos) {}

  bool scalable() { return false; }
};

class intExp : public literalExp {
  int value;

public:
  intExp(position pos, int value)
    : literalExp(pos), value(value) {}

  void prettyprint(ostream &out, int indent);

  types::ty *trans(coenv &e);
  types::ty *getType(coenv &) { return types::primInt(); }
};

class realExp : public literalExp {
protected:
  double value;

public:
  realExp(position pos, double value)
    : literalExp(pos), value(value) {}

  void prettyprint(ostream &out, int indent);

  types::ty *trans(coenv &e);
  types::ty *getType(coenv &) { return types::primReal(); }
};


class stringExp : public literalExp {
  mem::string str;

public:
  stringExp(position pos, string str)
    : literalExp(pos), str(str) {}

  void prettyprint(ostream &out, int indent);

  types::ty *trans(coenv &e);
  types::ty *getType(coenv &) { return types::primString(); }
};

class booleanExp : public literalExp {
  bool value;

public:
  booleanExp(position pos, bool value)
    : literalExp(pos), value(value) {}

  void prettyprint(ostream &out, int indent);

  types::ty *trans(coenv &e);
  types::ty *getType(coenv &) { return types::primBoolean(); }
};

class newPictureExp : public literalExp {

public:
  newPictureExp(position pos)
    : literalExp(pos) {}

  void prettyprint(ostream &out, int indent);

  types::ty *trans(coenv &e);
  types::ty *getType(coenv &) { return types::primPicture(); }
};

class nullPathExp : public literalExp {

public:
  nullPathExp(position pos)
    : literalExp(pos) {}

  void prettyprint(ostream &out, int indent);

  types::ty *trans(coenv &e);
  types::ty *getType(coenv &) { return types::primPath(); }
};

class nullExp : public literalExp {

public:
  nullExp(position pos)
    : literalExp(pos) {}

  void prettyprint(ostream &out, int indent);

  types::ty *trans(coenv &e);
  types::ty *getType(coenv &) { return types::primNull(); }
};

class quoteExp : public exp {
  runnable *value;

public:
  quoteExp(position pos, runnable *value)
    : exp(pos), value(value) {}

  void prettyprint(ostream &out, int indent);

  types::ty *trans(coenv &e);
  types::ty *getType(coenv &) { return types::primCode(); }
};

// A list of expressions used in a function call.
class explist : public absyn {
  typedef mem::vector<exp *> expvector;
  expvector exps;

public:
  explist(position pos)
    : absyn(pos) {}

  virtual ~explist() {}
  
  virtual void add(exp *e) {
    exps.push_back(e);
  }

  virtual void prettyprint(ostream &out, int indent);

  virtual size_t size() {
    return exps.size();
  }
  
  virtual exp * operator[] (size_t index) {
    return exps[index];
  }
};

struct argument {
  exp *val;
  symbol *name;

#if 0
  argument(exp *val=0, symbol *name=0)
    : val(val), name(name) {}
#endif

  void prettyprint(ostream &out, int indent);

  // Tests if a named argument could be mistaken for an assignment, and
  // prints a warning if so.
  void assignAmbiguity(coenv &e);
};

class arglist : public gc {
public:
  typedef mem::vector<argument> argvector;
  argvector args;
  argument rest;

  arglist()
    : args(), rest() {}

  virtual ~arglist() {}
  
  virtual void addFront(argument a) {
    args.insert(args.begin(), a);
  }

  virtual void addFront(exp *val, symbol *name=0) {
    argument a; a.val=val; a.name=name;
    addFront(a);
  }

  virtual void add(argument a) {
    args.push_back(a);
  }

  virtual void add(exp *val, symbol *name=0) {
    argument a; a.val=val; a.name=name;
    add(a);
  }

  virtual void prettyprint(ostream &out, int indent);

  virtual size_t size() {
    return args.size();
  }
  
  virtual argument operator[] (size_t index) {
    return args[index];
  }

  virtual argument getRest() {
    return rest;
  }
};


class callExp : public exp {
protected:
  exp *callee;
  arglist *args;

private:
  // Cache the application when it's determined.
  application *ca;

  // Warns of ambiguity with assign expression in named arguments.
  void argAmbiguity(coenv &e);

  types::signature *argTypes(coenv& e);
  application *resolve(coenv &e,
                       types::overloaded *o,
                       types::signature *source);
  void reportMismatch(symbol *s,
                      types::function *ft,
                      types::signature *source);
  application *getApplication(coenv &e);

public:
  callExp(position pos, exp *callee, arglist *args)
    : exp(pos), callee(callee), args(args), ca(0) { assert(args); }

  callExp(position pos, exp *callee)
    : exp(pos), callee(callee), args(new arglist()), ca(0) {}

  callExp(position pos, exp *callee, exp *arg1)
    : exp(pos), callee(callee), args(new arglist()), ca(0) {
      args->add(arg1);
    }

  callExp(position pos, exp *callee, exp *arg1, exp *arg2)
    : exp(pos), callee(callee), args(new arglist()), ca(0) {
      args->add(arg1);
      args->add(arg2);
    }

  callExp(position pos, exp *callee, exp *arg1, exp *arg2, exp *arg3)
    : exp(pos), callee(callee), args(new arglist()), ca(0) {
      args->add(arg1);
      args->add(arg2);
      args->add(arg3);
    }

  void prettyprint(ostream &out, int indent);

  types::ty *trans(coenv &e);
  types::ty *getType(coenv &e);
};

class pairExp : public exp {
  exp *x;
  exp *y;

public:
  pairExp(position pos, exp *x, exp *y)
    : exp(pos), x(x), y(y) {}

  void prettyprint(ostream &out, int indent);

  types::ty *trans(coenv &e);
  types::ty *getType(coenv &) { return types::primPair(); }
};

class tripleExp : public exp {
  exp *x;
  exp *y;
  exp *z;

public:
  tripleExp(position pos, exp *x, exp *y, exp *z)
    : exp(pos), x(x), y(y), z(z) {}

  void prettyprint(ostream &out, int indent);

  types::ty *trans(coenv &e);
  types::ty *getType(coenv &) { return types::primTriple(); }
};

class transformExp : public exp {
  exp *x;
  exp *y;
  exp *xx,*xy,*yx,*yy;

public:
  transformExp(position pos, exp *x, exp *y, exp *xx, exp *xy, exp *yx,
	       exp *yy)
    : exp(pos), x(x), y(y), xx(xx), xy(xy), yx(yx), yy(yy) {}

  void prettyprint(ostream &out, int indent);

  types::ty *trans(coenv &e);
  types::ty *getType(coenv &) { return types::primTransform(); }
};

class castExp : public exp {
  ty *target;
  exp *castee;

  types::ty *tryCast(coenv &e, types::ty *t, types::ty *s,
                     symbol *csym);
public:
  castExp(position pos, ty *target, exp *castee)
    : exp(pos), target(target), castee(castee) {}

  void prettyprint(ostream &out, int indent);

  types::ty *trans(coenv &e);
  types::ty *getType(coenv &e);
};

class nullaryExp : public callExp {
public:
  nullaryExp(position pos, symbol *op)
    : callExp(pos, new nameExp(pos, op)) {}
};

class unaryExp : public callExp {
public:
  unaryExp(position pos, exp *base, symbol *op)
    : callExp(pos, new nameExp(pos, op), base) {}
};

class binaryExp : public callExp {
public:
  binaryExp(position pos, exp *left, symbol *op, exp *right)
    : callExp(pos, new nameExp(pos, op), left, right) {}
};

// Scaling expressions such as 3sin(x).
class scaleExp : public binaryExp {
  exp *getLeft() {
    return (*this->args)[0].val;
  }
  exp *getRight() {
    return (*this->args)[1].val;
  }
public:
  scaleExp(position pos, exp *left, exp *right)
    : binaryExp(pos, left, symbol::trans("*"), right) {}

  void prettyprint(ostream &out, int indent);

  types::ty *trans(coenv &e);
  //types::ty *getType(coenv &e);

  bool scalable() { return false; }
};

// Used for tension, which takes two real values, and a boolean to denote if it
// is a tension atleast case.
class ternaryExp : public callExp {
public:
  ternaryExp(position pos, exp *left, symbol *op, exp *right, exp *last)
    : callExp(pos, new nameExp(pos, op), left, right, last) {}
};

// The a ? b : c ternary operator.
class conditionalExp : public exp {
  exp *test;
  exp *onTrue;
  exp *onFalse;

public:
  conditionalExp(position pos, exp *test, exp *onTrue, exp *onFalse)
    : exp(pos), test(test), onTrue(onTrue), onFalse(onFalse) {}

  void prettyprint(ostream &out, int indent);

  void transToType(coenv &e, types::ty *target);
  types::ty *trans(coenv &e);
  types::ty *getType(coenv &e);
  
};
 
class andOrExp : public exp {
protected:
  exp *left;
  symbol *op;
  exp *right;

public:
  andOrExp(position pos, exp *left, symbol *op, exp *right)
    : exp(pos), left(left), op(op), right(right) {}

  types::ty *trans(coenv &e);
  types::ty *getType(coenv &);

  virtual types::ty *baseTrans(coenv &e) = 0;
  virtual types::ty *baseGetType(coenv &) {
    return types::primBoolean();
  }
};

class orExp : public andOrExp {
public:
  orExp(position pos, exp *left, symbol *op, exp *right)
    : andOrExp(pos, left, op, right) {}

  void prettyprint(ostream &out, int indent);

  types::ty *baseTrans(coenv &e);
};

class andExp : public andOrExp {
public:
  andExp(position pos, exp *left, symbol *op, exp *right)
    : andOrExp(pos, left, op, right) {}

  void prettyprint(ostream &out, int indent);

  types::ty *baseTrans(coenv &e);
};

class joinExp : public callExp {
public:
  joinExp(position pos, symbol *op)
    : callExp(pos, new nameExp(pos, op)) {}

  void pushFront(exp *e) {
    args->addFront(e);
  }
  void pushBack(exp *e) {
    args->add(e);
  }

  void prettyprint(ostream &out, int indent);
};

class specExp : public exp {
  symbol *op;
  exp *arg;
  camp::side s;

public:
  specExp(position pos, symbol *op, exp *arg, camp::side s=camp::OUT)
    : exp(pos), op(op), arg(arg), s(s) {}

  void setSide(camp::side ss) {
    s=ss;
  }

  void prettyprint(ostream &out, int indent);

  types::ty *trans(coenv &e);
  types::ty *getType(coenv &e);
};

class assignExp : public exp {
protected:
  exp *dest;
  exp *value;

  // This is basically a hook to facilitate selfExp.  dest is given as an
  // argument since it will be a temporary in translation in order to avoid
  // multiple evaluation.
  virtual exp *ultimateValue(exp *) {
    return value;
  }

public:
  assignExp(position pos, exp *dest, exp *value)
    : exp(pos), dest(dest), value(value) {}

  void prettyprint(ostream &out, int indent);

  void transAsType(coenv &e, types::ty *target);
  types::ty *trans(coenv &e);
  types::ty *getType(coenv &e);
};

class selfExp : public assignExp {
  symbol *op;

  exp *ultimateValue(exp *dest) {
    return new binaryExp(getPos(), dest, op, value);
  }

public:
  selfExp(position pos, exp *dest, symbol *op, exp *value)
    : assignExp(pos, dest, value), op(op) {}

  void prettyprint(ostream &out, int indent);
};

class prefixExp : public exp {
  exp *dest;
  symbol *op;

public:
  prefixExp(position pos, exp *dest, symbol *op)
    : exp(pos), dest(dest), op(op) {}

  void prettyprint(ostream &out, int indent);

  bool scalable() { return false; }

  types::ty *trans(coenv &e);
  types::ty *getType(coenv &e);
};

// Postfix expresions are illegal. This is caught here as we can give a
// more meaningful error message to the user, rather than a "parse
// error."
class postfixExp : public exp {
  exp *dest;
  symbol *op;

public:
  postfixExp(position pos, exp *dest, symbol *op)
    : exp(pos), dest(dest), op(op) {}

  void prettyprint(ostream &out, int indent);

  types::ty *trans(coenv &e);
  types::ty *getType(coenv &) { return types::primError(); }
};
  
} // namespace absyntax

#endif
