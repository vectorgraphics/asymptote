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

#include <list>

#include "types.h"
#include "symbol.h"
#include "absyn.h"
#include "name.h"
#include "guideflags.h"

namespace trans {
class coenv;
}

namespace as {

using std::list;
using trans::coenv;
using sym::symbol;
using vm::inst;
using types::record;
using types::array;

class varinit : public absyn {
public:
  varinit(position pos)
    : absyn(pos) {}

  // This determines what instruction and needed to put the associated
  // value onto the stack, then adds those instructions to the current
  // lambda in e.
  // In some expressions and initializers, the target type needs to be
  // known in order to translate properly.  For most expressions, this is
  // kept to a minimum.
  virtual void trans(coenv &e, types::ty *target) = 0;
};

class arrayinit : public varinit {
  list<varinit *> inits;

public:
  arrayinit(position pos)
    : varinit(pos) {}

  virtual ~arrayinit() 
    {}

  void prettyprint(ostream &out, int indent);

  void trans(coenv &e, types::ty *target);

  void add(varinit *init) {
    inits.push_back(init);
  }
};

class exp : public varinit {
public:
  exp(position pos)
    : varinit(pos) {}

  void prettyprint(ostream &out, int indent);

  // When reporting errors with function calls, it is nice to say "no
  // functon f(int)" instead of "no function matching signature
  // (int)."  Hence, this method returns the name of the expression if
  // there is one.
  virtual symbol *getName()
  {
    return 0;
  }

  // Checks if this exp can be used as a statement on its own.
  virtual bool stmable() { return false; }

  // Checks if expression can be used as the right side of a scale
  // expression.  ie. 3sin(x)
  virtual bool scalable() { return false; }
  
  // Translates the expression to the given target type.  The default
  // behavior is to trans without the target, then perform a cast. 
  void trans(coenv &, types::ty *target);

  // Translates the expression and returns the resultant type.
  // For some expressions, this will be ambiguous and return an error.
  virtual types::ty *trans(coenv &) = 0;

  // Figures out the type of the expression without translating the code
  // into the virtual machine language or reporting errors to em.
  // This must follow a few rules to ensure proper translation:
  //   1. If this returns a valid type, t, trans(e) must return t or
  //      report an error, and trans(e, t) must run either reporting an
  //      error or reporting no error and yielding the same result.
  //   2. If this returns a superposition of types (ie. for overloaded
  //      functions), trans must not return a singular type, and every
  //      type in the superposition must run without error properly
  //      if fed to trans(e, t).
  //   3. If this returns ty_error, then so must a call to trans(e) and
  //      any call to either trans must report an error to em.
  //   4. Any call to trans(e, t) with a type that is not returned by
  //      getType() (or one of the subtypes in case of a superposition)
  //      or any type not implicitly castable from the above must report an
  //      error.
  virtual types::ty *getType(coenv &) { return types::primError(); }

  // The expression is being used as an address to write to.
  virtual void transWrite(coenv &, types::ty *) {
    em->error(getPos());
    *em << "expression cannot be used as an address";
  }

  virtual void transCall(coenv &e, types::ty *target);
};

class nameExp : public exp {
  name *value;

public:
  nameExp(position pos, name *value)
    : exp(pos), value(value) {}

  void prettyprint(ostream &out, int indent);

  symbol *getName()
  {
    return value->getName();
  }

  bool scalable() { return true; }

  void trans(coenv &e, types::ty *target) {
    value->varTrans(e, target);
  }

  types::ty *trans(coenv &e) {
    types::ty *t = value->varGetType(e);
    if (t->kind == types::ty_overloaded) {
      em->error(getPos());
      *em << "use of variable \'" << *value << "\' is ambiguous";
      return types::primError();
    }
    else {
      value->varTrans(e, t);
      return t;
    }
  }

  types::ty *getType(coenv &e) {
    return value->varGetType(e);
  }

  void transWrite(coenv &e, types::ty *target) {
    value->varTransWrite(e, target);
  }
  
  void transCall(coenv &e, types::ty *target) {
    value->varTransCall(e, target);
  }
};

// Most fields accessed are handled as parts of qualified names, but in cases
// like f().x or (new t).x, a separate expression is needed.
class fieldExp : public exp {
  exp *object;
  symbol *field;

  types::ty *getObject(coenv& e);
  record *getRecord(types::ty *qt);
  record *transRecord(coenv& e, types::ty *qt);

public:
  fieldExp(position pos, exp *object, symbol *field)
    : exp(pos), object(object), field(field) {}

  void prettyprint(ostream &out, int indent);

  symbol *getName()
  {
    return field;
  }

  // This has the whole smorgasbord of trans functions!
  void trans(coenv &e, types::ty *target);
  types::ty *trans(coenv &e);
  types::ty *getType(coenv &e);
  void transWrite(coenv &e, types::ty *target);
  void transCall(coenv &e, types::ty *target);
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
// Exceptional expressions such as 3sin(x).
class scaleExp : public exp {
  exp *left;
  exp *right;

public:
  scaleExp(position pos, exp *left, exp *right)
    : exp(pos), left(left), right(right) {}

  void prettyprint(ostream &out, int indent);

  types::ty *trans(coenv &e);
  types::ty *getType(coenv &e);
};

class intExp : public exp {
  int value;

public:
  intExp(position pos, int value)
    : exp(pos), value(value) {}

  void prettyprint(ostream &out, int indent);

  types::ty *trans(coenv &e);
  types::ty *getType(coenv &) { return types::primInt(); }
};

class realExp : public exp {
protected:
  double value;

public:
  realExp(position pos, double value)
    : exp(pos), value(value) {}

  void prettyprint(ostream &out, int indent);

  types::ty *trans(coenv &e);
  types::ty *getType(coenv &) { return types::primReal(); }
};


class stringExp : public exp {
  std::string str;

public:
  stringExp(position pos, std::string str)
    : exp(pos), str(str) {}

  void prettyprint(ostream &out, int indent);

  types::ty *trans(coenv &e);
  types::ty *getType(coenv &) { return types::primString(); }
};

class booleanExp : public exp {
  bool value;

public:
  booleanExp(position pos, bool value)
    : exp(pos), value(value) {}

  void prettyprint(ostream &out, int indent);

  types::ty *trans(coenv &e);
  types::ty *getType(coenv &) { return types::primBoolean(); }
};

class nullPictureExp : public exp {

public:
  nullPictureExp(position pos)
    : exp(pos) {}

  void prettyprint(ostream &out, int indent);

  types::ty *trans(coenv &e);
  types::ty *getType(coenv &) { return types::primPicture(); }
};

class nullPathExp : public exp {

public:
  nullPathExp(position pos)
    : exp(pos) {}

  void prettyprint(ostream &out, int indent);

  types::ty *trans(coenv &e);
  types::ty *getType(coenv &) { return types::primPath(); }
};

class nullExp : public exp {

public:
  nullExp(position pos)
    : exp(pos) {}

  void prettyprint(ostream &out, int indent);

  types::ty *trans(coenv &e);
  types::ty *getType(coenv &) { return types::primNull(); }
};

// A list of expressions used in a function call.
class explist : public absyn {
  std::vector<exp *> exps;

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
  
  virtual types::ty *trans(coenv &e, int index);
  virtual void trans(coenv &e, types::ty *target, int index);
  virtual types::ty *getType(coenv &e, int index);
};


class callExp : public exp {
  exp *callee;
  explist *args;

public:
  callExp(position pos, exp *callee, explist *args)
    : exp(pos), callee(callee), args(args) {}

  void prettyprint(ostream &out, int indent);

  // A function call can be used alone in a statement.
  bool stmable() { return true; }

  bool scalable() { return true; }
  
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

  bool scalable() { return true; }

  types::ty *trans(coenv &e);
  types::ty *getType(coenv &) { return types::primPair(); }
};

class unaryExp : public exp {
  exp *base;
  symbol *op;

public:
  unaryExp(position pos, exp *base, symbol *op)
    : exp(pos), base(base), op(op) {}

  void prettyprint(ostream &out, int indent);

  types::ty *trans(coenv &e);
  types::ty *getType(coenv &e);
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

class castExp : public exp {
  name *target;
  exp *castee;

public:
  castExp(position pos, name *target, exp *castee)
    : exp(pos), target(target), castee(castee) {}

  void prettyprint(ostream &out, int indent);

  types::ty *trans(coenv &e);
  types::ty *getType(coenv &e);
};

class binaryExp : public exp {
  exp *left;
  symbol *op;
  exp *right;

public:
  binaryExp(position pos, exp *left, symbol *op, exp *right)
    : exp(pos), left(left), op(op), right(right) {}

  void prettyprint(ostream &out, int indent);

  bool scalable() { return true; }

  // We may need to re-implement this function.
  //void trans(coenv &e, types::ty *target);

  types::ty *trans(coenv &e);
  types::ty *getType(coenv &e);
  
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

  // NOTE: decide if this is scalable.

  void trans(coenv &e, types::ty *target);
  types::ty *trans(coenv &e);
  types::ty *getType(coenv &e);
  
};
 

// dir refers to the {} direction specifiers before or after a knot.
class dir : public absyn {
public:
  dir(position pos)
    : absyn(pos) {}

  virtual void trans(coenv &e) = 0;
 
  // What flags to mark in a joinExp.
  virtual int leftFlags()
    { return 0; }
  virtual int rightFlags()
    { return 0; }
}; 

class givenDir : public dir {
  exp *base;

public:
  givenDir(position pos, exp *base)
    : dir(pos), base(base) {}

  void prettyprint(ostream &out, int indent);

  void trans(coenv &e);

  int leftFlags()
    { return run::LEFT_GIVEN; }
  int rightFlags()
    { return run::RIGHT_GIVEN; }
};

class curlDir : public dir {
  exp *base;

public:
  curlDir(position pos, exp *base)
    : dir(pos), base(base) {}

  void prettyprint(ostream &out, int indent);

  void trans(coenv &e);

  int leftFlags()
    { return run::LEFT_CURL; }
  int rightFlags()
    { return run::RIGHT_CURL; }
};
  
// join refers to the section between knots, including the tension,
// controls, and direction specifiers
class join : public absyn{
  position pos;

  dir *leftDir;
  dir *rightDir;

  // May be tensions or controls.
  exp *leftCont;
  exp *rightCont;

  bool tension;
  bool atleast;

public:
  join(position pos, exp *leftCont, exp *rightCont, bool tension, bool atleast = false)
    : absyn(pos), leftDir(0), rightDir(0),
      leftCont(leftCont), rightCont(rightCont),
      tension(tension), atleast(atleast) {}
  join(position pos, exp *leftCont, bool tension, bool atleast = false)
    : absyn(pos), leftDir(0), rightDir(0),
      leftCont(leftCont), rightCont(0),
      tension(tension), atleast(atleast) {}
  join(position pos)
    : absyn(pos), leftDir(0), rightDir(0),
      leftCont(0), rightCont(0),
      tension(false), atleast(false) {}

  virtual void setLeftDir(dir *leftDir)
    { this->leftDir = leftDir; }
  virtual void setRightDir(dir *rightDir)
    { this->rightDir = rightDir; }

  virtual void prettyprint(ostream &out, int indent);

  virtual void trans(coenv &e);
};

class joinExp : public exp {
  exp *left;

  join *middle;

  exp *right;

public:
  joinExp(position pos, exp *left, join *middle, exp *right)
    : exp(pos), left(left), middle(middle), right(right) {}

  void prettyprint(ostream &out, int indent);

  types::ty *trans(coenv &e);
  types::ty *getType(coenv &) { return types::primGuide(); }
};

// This expression is just a placeholder for the CYCLE keyword which can
// be thought of as a guide on its own.
class cycleExp : public exp {
public:
  cycleExp(position pos)
    : exp(pos) {}

  void prettyprint(ostream &out, int indent);

  types::ty *trans(coenv &e);
  types::ty *getType(coenv &) { return types::primGuide(); }
};

// This handles guide expression with a direction specifier tagged on to
// the end, ie. a..b..c{}
class dirguideExp : public exp {
  exp *base;

  dir *tag;

public:
  dirguideExp(position pos, exp *base, dir *tag)
    : exp(pos), base(base), tag(tag) {}

  void prettyprint(ostream &out, int indent);

  types::ty *trans(coenv &e);
  types::ty *getType(coenv &) { return types::primGuide(); }
};

class assignExp : public exp {
  exp *dest;
  exp *value;

public:
  assignExp(position pos, exp *dest, exp *value)
    : exp(pos), dest(dest), value(value) {}

  void prettyprint(ostream &out, int indent);

  bool stmable() { return true; }

  void trans(coenv &e, types::ty *target);
  types::ty *trans(coenv &e);
  types::ty *getType(coenv &e);
};

class selfExp : public exp {
  exp *dest;
  symbol *op;
  exp *value;

public:
  selfExp(position pos, exp *dest, symbol *op, exp *value)
    : exp(pos), dest(dest), op(op), value(value) {}

  void prettyprint(ostream &out, int indent);

  bool stmable() { return true; }
  
  types::ty *trans(coenv &e);
  types::ty *getType(coenv &e);
};

class prefixExp : public exp {
  exp *dest;
  symbol *op;

public:
  prefixExp(position pos, exp *dest, symbol *op)
    : exp(pos), dest(dest), op(op) {}

  void prettyprint(ostream &out, int indent);

  bool stmable() { return true; }
  
  types::ty *trans(coenv &e);
  types::ty *getType(coenv &e);
};

// Postfix expresions are illegal. This is caught here as we can give a
// more meaningmore error message to the user, rather than a "parse
// error."
class postfixExp : public exp {
  exp *dest;
  symbol *op;

public:
  postfixExp(position pos, exp *dest, symbol *op)
    : exp(pos), dest(dest), op(op) {}

  void prettyprint(ostream &out, int indent);

  bool stmable() { return true; }
  
  types::ty *trans(coenv &e);
  types::ty *getType(coenv &) { return types::primError(); }
};
  
// Global array of default expressions
extern vector<varinit *> defaultExp;
  
} // namespace as

#endif
