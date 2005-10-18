/*****
 * genv.cc
 * Andy Hammerlindl 2002/08/29
 *
 * This is the global environment for the translation of programs.  In
 * actuality, it is basically a module manager.  When a module is
 * requested, it looks for the corresponding filename, and if found,
 * parses and translates the file, returning the resultant module.
 *
 * genv sets up the basic type bindings and function bindings for
 * builtin functions, casts and operators, and imports plain (if set),
 * but all other initialization, is done by the local environmet defined
 * in env.h.
 *****/

#include <sstream>
#include <unistd.h>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "genv.h"
#include "env.h"
#include "dec.h"
#include "stm.h"
#include "types.h"
#include "settings.h"
#include "builtin.h"
#include "runtime.h"
#include "parser.h"
#include "locate.h"
#include "interact.h"

using namespace types;

namespace trans {

record *genv::loadModule(symbol *id, std::string filename) {
  // Get the abstract syntax tree.
  absyntax::file *ast = parser::parseFile(filename);
  em->sync();
  
  // Create the new module.
  record *r = new record(id, new frame(0,0));

  // Create coder and environment to translate the module.
  // File-level modules have dynamic fields by default.
  coder c(r, 0);
  env e(*this);
  coenv ce(c, e);

  // Translate the abstract syntax.
  ast->transAsRecordBody(ce, r);
  em->sync();

  // NOTE: Move this to a similar place as settings::translate.
  if(settings::listonly)
    r->e.list();
  
  return r;
}


record *genv::getModule(symbol *id, std::string filename) {
  record *r=imap[filename];
  if (r)
    return r;
  else
    return imap[filename]=loadModule(id, filename);
}

typedef vm::stack::importInitMap importInitMap;

importInitMap *genv::getInitMap()
{
  struct initMap : public importInitMap {
    genv &ge;
    initMap(genv &ge)
      : ge(ge) {}
    lambda *operator[](mem::string s) {
      record *r=ge.imap[s];
      return r ? r->getInit() : 0;
    }
  };
  
  return new initMap(*this);
}

} // namespace trans
