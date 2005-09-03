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
#include "fundec.h"

namespace absyntax {

typedef fundef newFunctionExp;

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
