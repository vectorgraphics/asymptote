/*****
 * genv.h
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

#ifndef GENV_H
#define GENV_H

#include "memory.h"
#include "table.h"
#include "record.h"
#include "absyn.h"
#include "access.h"
#include "coenv.h"
#include "stack.h"

using types::record;
using vm::lambda;

namespace trans {

class genv : public gc {
  // The initializer functions for imports, indexed by filename.
  typedef mem::map<const std::string,record *> importMap;
  importMap imap;

  // List of modules in translation.  Used to detect and prevent infinite
  // recursion in loading modules.
  mem::list<std::string> inTranslation;

  // Checks for recursion in loading, reporting an error and throwing an
  // exception if it occurs.
  void checkRecursion(std::string filename);

  // Translate a module to build the record type.
  record *loadModule(symbol *name, std::string s);

public:
  genv();

  // Get an imported module, translating if necessary.
  record *getModule(symbol *name, std::string s);

  // Uses the filename->record map to build a filename->initializer map to be
  // used at runtime.
  vm::stack::importInitMap *getInitMap();
  
  void listValues(const std::string name, record* &vals) {
    vals->e.list();
  }
  
  void list() {
    for(importMap::iterator i = imap.begin(); i != imap.end(); ++i)
      listValues(i->first, i->second);
  }
  
};

} // namespace trans

#endif
