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

#include "common.h"
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
  typedef mem::map<symbol,record *> importMap;
  importMap imap;

  // List of modules in translation.  Used to detect and prevent infinite
  // recursion in loading modules.
  mem::list<string> inTranslation;

  // Checks for recursion in loading, reporting an error and throwing an
  // exception if it occurs.
  void checkRecursion(string filename);

  // Translate a module to build the record type.
  record *loadModule(symbol name, string s);
  record *loadTemplatedModule(
      symbol id,
      string filename,
      mem::vector<absyntax::namedTyEntry*> *args,
      coenv& e
  );

public:
  genv();

  // Get an imported module, translating if necessary.
  record *getModule(symbol name, string filename);
  record *getTemplatedModule(
      symbol index,
      string filename,
      mem::vector<absyntax::namedTyEntry*> *args,
      coenv& e
  );
  record *getLoadedModule(symbol index);

  // Uses the filename->record map to build a filename->initializer map to be
  // used at runtime.
  vm::stack::importInitMap *getInitMap();
};

/* Plan for implementation of templated modules:
 *    
 *    Translating an access declaration:
 *    
 *    access Map(Key=A, Value=B) as MapAB;
 *    
 *    run encodeLevel for both A and B
 *    this should give the parent records for each struct
 *    encode pushing the string "Map/1234567" on the stack
 *    encode call to builtin loadTemplatedModule
 *    also save into MapAB (varinit)
 *    
 *    build list of types (or tyEntry?)
 *    
 *    also ensure names match
 *    
 *    *****
 *    
 *    At runtime, loadTemplatedModule pops the string
 *    
 *    if the module is already loaded, it pops the levels
 *    and returns the already loaded module.
 *    
 *    if the module is not loaded, it leaves the levels on the stack
 *    and calls the initializer for the templated module
 *    
 *    it might be easiest to give the number of pushed params as an argument
 *    to loadTemplatedModule (ints and strings have no push/pop)
 *    
 *    *****
 *    
 *    Translating a templated module
 *    
 *    we start translating a file with a list of (name, tyEntry)
 *    or (name, type) pairs
 *    
 */

} // namespace trans

#endif
