/*****
 * newexp.h
 * Andy Hammerlindl 2003/07/28
 *
 * Handles the abstract syntax for expressions the create new objects,
 * such as record, array, and function constructors.
 *****/

#ifndef NEWEXP_H
#define NEWEXP_H

#include "exp.h"
#include "dec.h"

namespace absyn {

class newFunctionExp : public exp {
  ty *result;
  formals *params;
  stm *body;

public:
  newFunctionExp(position pos, ty *result, formals *params, stm *body)
    : exp(pos), result(result), params(params), body(body) {}

  types::ty *trans(coenv &e);
  types::ty *getType(coenv &e);

  bool stmable() 
  {
    return true;
  }
};

class newRecordExp : public exp {
  ty *result;

public:
  newRecordExp(position pos, ty *result)
    : exp(pos), result(result) {}

  types::ty *trans(coenv &e);
  types::ty *getType(coenv &e);

  bool stmable()
  {
    return true;
  }
};
  
class newArrayExp : public exp {
  ty *celltype;
  explist *dimexps;
  dimensions *dims;
  arrayinit *ai;

public:
  newArrayExp(position pos,
              ty *celltype,
	      explist *dimexps,
	      dimensions *dims,
	      arrayinit *ai)
    : exp(pos), celltype(celltype), dimexps(dimexps), dims(dims), ai(ai) {}

  void prettyprint(ostream &out, int indent);

  types::ty *trans(coenv &e);
  types::ty *getType(coenv &e);

  bool stmable()
  {
    return true;
  }
};
  
} // namespace absyn

#endif
