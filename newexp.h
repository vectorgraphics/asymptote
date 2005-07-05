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

namespace absyntax {

class newFunctionExp : public exp {
  fundef fun;

public:
  newFunctionExp(position pos, ty *result, formals *params, stm *body)
    : exp(pos), fun(pos, result, params, body) {}

  types::ty *trans(coenv &e) {
    fun.trans(e);
    return cgetType(e);
  }

  types::ty *getType(coenv &e) {
    return fun.getType(e, true);
  }
};

class newRecordExp : public exp {
  ty *result;

public:
  newRecordExp(position pos, ty *result)
    : exp(pos), result(result) {}

  types::ty *trans(coenv &e);
  types::ty *getType(coenv &e);
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
};
  
} // namespace absyntax

#endif
