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

#include "cast.h"
#include "table.h"
#include "record.h"
#include "absyn.h"
#include "access.h"
#include "coenv.h"

using types::record;
using vm::lambda;

namespace trans {

class genv {
  // The collection of loaded modules.
  sym::table<record *> modules;

  // Base environment for allocating global variables and encoding
  // static code of file-level modules.
  friend class env;
  coder base_coder;
  env base_env;
  coenv base_coenv;
  
  tenv te;
  venv ve;
  menv me;
public:
  genv();

  void autoloads(const std::string& outname);
  void loadPlain();
  void loadGUI(const std::string& outname);

  // If a module is already loaded, this will return it.  Otherwise, it
  // returns null.
  record *getModule(symbol *id);

  // Loads a module from the corresponding file and adds it to the table
  // of loaded modules.  If a module of the same name was already
  // loaded, it will be shadowed by the new one.
  // If the module could not be loaded, returns null.
  record *loadModule(symbol *id, absyntax::file *ast = 0);

  // Returns a function that statically initializes all loaded modules.
  // Then runs the dynamic initializer of r.
  // This should be the lowest-level function run by the stack.
  // loadModule() should not be called after calling this function.
  lambda *bootupModule(record *r);
};

} // namespace trans

#endif
