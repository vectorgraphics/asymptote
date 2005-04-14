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
#include <string>

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

// Whether the module name should be visible like an import when translating
// that module.
#define SELF_IMPORT 1

using namespace std;
using namespace types;
using vm::inst;

namespace trans {

genv::genv()
 : base_coder(),
   base_env(*this),
   base_coenv(base_coder,base_env)
{
  types::initializeCasts();
  types::initializeInitializers();

  base_tenv(te);
  base_venv(ve);
  base_menv(me);
}

void genv::loadPlain()
{
  static absyntax::importdec iplain(position::nullPos(),
                                    symbol::trans("plain"));
  iplain.trans(base_coenv);
  me.beginScope(); // NOTE: This is unmatched.
}

void genv::loadGUI(string outname) 
{
  string GUIname=buildname(outname,"gui");
  std::ifstream exists(GUIname.c_str());
  if(exists) {
    if(settings::clearGUI) unlink(GUIname.c_str());
    else {
      absyntax::importdec igui(position::nullPos(),
                               symbol::trans(GUIname.c_str()));
      igui.trans(base_coenv);
      me.beginScope(); // NOTE: This is unmatched.
    }
  }
}

// If a module is already loaded, this will return it.  Otherwise, it
// returns null.
record *genv::getModule(symbol *id)
{
  return modules.look(id);
}

// Loads a module from the corresponding file and adds it to the table
// of loaded modules.  If a module of the same name was already
// loaded, it will be shadowed by the new one.
// If the module could not be loaded, returns null.
record *genv::loadModule(symbol *id, absyntax::file *ast)
{
  // Get the abstract syntax tree.
  if (ast == 0) ast = parser::parseFile(*id);
  em->sync();

  if (!ast)
    return 0;

  //ast->prettyprint(stdout, 0);
 
  // Create the new module.
  record *r = base_coder.newRecord(id);

  // Add it to the table of modules.
  modules.enter(id, r);

  // Create coder and environment to translate the module.
  // File-level modules have dynamic fields by default.
  coder c=base_coder.newRecordInit(r);
  coenv e(c, base_env);

  // Make the record name visible like an import when translating the module.
#if SELF_IMPORT
  e.e.beginScope();
  import i(r, c.thisLocation());
  e.e.enterImport(id, &i);
#endif

  // Translate the abstract syntax.
  ast->transAsRecordBody(e, r);
  em->sync();

#if SELF_IMPORT
  e.e.endScope();
#endif

  return r;
}

// Returns a function that statically initializes all loaded modules.
// Then runs the dynamic initializer of r.
// This should be the lowest-level function run by the stack.
lambda *genv::bootupModule(record *r)
{
  // Encode the record dynamic instantiation.
  if (!base_coder.encode(r->getLevel()->getParent())) {
    em->compiler();
    *em << "invalid bootup structure";
    em->sync();
    return 0;
  }


  // Encode the allocation.
  inst i; i.op = inst::makefunc; i.lfunc = r->getInit();
  base_coder.encode(i);
  base_coder.encode(inst::popcall);
  base_coder.encode(inst::pop);

  base_coder.encode(inst::builtin, run::exitFunction);
  
  // Return the finished function.
  return base_coder.close();
}

} // namespace trans
