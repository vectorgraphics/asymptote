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

using namespace types;
using trans::import;
using trans::coder;
using trans::coenv;

namespace absyntax {

types::ty *newFunctionExp::trans(coenv &e)
{
  // Check for illegal default values.
  params->reportDefaults();
  
  // NOTE: Duplicate code with dec.cc
  
  // Create a new function environment.
  types::ty *rt = result->trans(e);
  function *ft = params->getType(rt, e);

  coder fc = e.c.newFunction(ft);
  coenv fe(fc,e.e);

  fe.e.beginScope();

  // Add the formal parameters to the environment.
  params->trans(fe);

  // Translate the body.
  body->trans(fe);

  if (rt->kind != ty_void &&
      rt->kind != ty_error &&
      !body->returns()) {
    em->error(body->getPos());
    *em << "function must return a value";
  }

  fe.e.endScope();

  // Use the lambda to put the function on the stack.
  lambda *l = fe.c.close();
  e.c.encode(inst::pushclosure);
  e.c.encode(inst::makefunc, l);

  //std::cout << "made new lambda:\n";
  //print(std::cout, l->code);

  return ft;
}

types::ty *newFunctionExp::getType(coenv &e)
{
  return params->getType(result->trans(e, true), e);
}

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
  inst i;
  i.op = inst::alloc;
  i.r = r->getRuntime();
  e.c.encode(i);

  return t;
}

types::ty *newRecordExp::getType(coenv &e)
{
  types::ty *t = result->trans(e, true);
  if (t->kind == ty_error)
    return t;
  else if (t->kind != ty_record)
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
    ai->trans(e, c);
    return c;
  } else if (dimexps || dims) {
    if (dimexps) {
      for (int i = 0; i < (int)dimexps->size(); ++i) {
        dimexps->trans(e, types::primInt(), i);
	c = new types::array(c);
      }
    }
    if (dims) {
      for (int i = 0; i < (int)dims->size(); ++i) {
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
