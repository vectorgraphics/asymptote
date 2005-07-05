/*****
 * newexp.cc
 * Andy Hammerlindl 2003/07/28
 *
 * Handles the abstract syntax for expressions the create new objects,
 * such as record, array, and function constructors.
 *****/

#include "newexp.h"
#include "stm.h"
#include "runtime.h"
#include "coenv.h"
#include "inst.h"

using namespace types;
using trans::import;
using trans::coder;
using trans::coenv;
using vm::inst;

namespace absyntax {

void printFrame(frame *f) {
  if (f == 0) {
    std::cerr << '0';
  }
  else {
    std::cerr << f << " of ";
    printFrame(f->getParent());
  }
}

types::ty *newRecordExp::trans(coenv &e)
{
  types::ty *t = result->trans(e);
  if (t->kind == ty_error)
    return t;
  else if (t->kind != ty_record) {
    em->error(getPos());
    *em << "type '" << *t << "' is not a structure";
    return primError();
  }

  // Put the enclosing frame on the stack.
  record *r = (record *)t;
  import *imp = result->getImport(e);
  if (imp) {
    // Put the import frame on the stack.
    imp->getLocation()->encodeRead(getPos(), e.c);

#if 0
    std::cerr << *(r->getName()) << ": ";
    printFrame(r->getLevel());
    std::cerr << std::endl;
    std::cerr << *(imp->getModule()->getName()) << ": ";
    printFrame(imp->getModule()->getLevel());
    std::cerr << std::endl;
#endif
   
    // Dig down to the record's parent's frame.
    if (!e.c.encode(r->getLevel()->getParent(),
		   imp->getModule()->getLevel())) {
      em->error(getPos());
      *em << "allocation of struct '" << *t << "' is not in a valid scope";
      return primError();
    }
  }
  else {
    if (!e.c.encode(r->getLevel()->getParent())) {
      em->error(getPos());
      *em << "allocation of struct '" << *t << "' is not in a valid scope";
      return primError();
    }
  }
 
  // Encode the allocation. 
  e.c.encode(inst::makefunc,r->getInit());
  e.c.encode(inst::popcall);

  return t;
}

types::ty *newRecordExp::getType(coenv &e)
{
  types::ty *t = result->trans(e, true);
  if (t->kind != ty_error && t->kind != ty_record)
    return primError();
  else
    return t;
}  

void newArrayExp::prettyprint(ostream &out, int indent)
{
  prettyname(out,"newArrayExp",indent);

  celltype->prettyprint(out, indent+1);
  if (dimexps) dimexps->prettyprint(out, indent+1);
  if (dims) dims->prettyprint(out, indent+1); 
  if (ai) ai->prettyprint(out, indent+1);
}

types::ty *newArrayExp::trans(coenv &e)
{
  types::ty *c = celltype->trans(e);
  if (c->kind == ty_void) {
    em->compiler(getPos());
    *em << "arrays cannot be of type void";
    return primError();
  }

  if (dims)
    c = dims->truetype(c);

  if (ai) {
    ai->transToType(e, c);
    return c;
  } else if (dimexps || dims) {
    if (dimexps) {
      for (size_t i = 0; i < dimexps->size(); ++i) {
        (*dimexps)[i]->transToType(e, types::primInt());
	c = new types::array(c);
      }
    }
    if (dims) {
      for (size_t i = 0; i < dims->size(); ++i) {
        e.c.encode(inst::intpush,0);
      }
    }
    e.c.encode(inst::intpush,
               (int) (dimexps ? dimexps->size():0
                      + dims ? dims->size():0));
    e.c.encode(inst::builtin, run::newDeepArray);

    return c;
  } else {
    em->compiler(getPos());
    *em << "new array expression must have either dims or dimexps";
    return primError();
  }
}

types::ty *newArrayExp::getType(coenv &e)
{
  types::ty *c = celltype->trans(e);
  if (c->kind == ty_void) {
    em->compiler(getPos());
    *em << "can't declare array of type void";
    return primError();
  }
  if (dims)
    c = dims->truetype(c);

  if (dimexps) {
    int depth = (int)dimexps->size();
    while (depth > 0) {
      c = new types::array(c);
      depth--;
    }
  }

  return c;
}


} // namespace absyntax
