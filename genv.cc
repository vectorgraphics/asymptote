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

// Whether the module name should be visible like an import when translating
// that module.
#define SELF_IMPORT 1

using namespace std;
using namespace types;
using vm::inst;

// The lexical analysis and parsing functions used by parseFile.
extern bool setlexer(std::string filename);
extern bool yyparse(void);
extern int yydebug;
extern int yy_flex_debug;

namespace trans {

// Adds the appropriate directory and suffix to the name.
string dirSymbolToFile(string s, symbol *id)
{
  ostringstream buf;
  if (!s.empty())
    buf << s << "/";
  
  size_t p=findextension((string)*id,settings::suffix);
  if(p < string::npos)
    buf << *id;
  else {
    p=findextension((string)*id,settings::guisuffix);
    if(p < string::npos)
      buf << *id;
    else
      buf << *id << "." << settings::suffix;
  }
  return buf.str();
}

bool exists(string filename)
{
  return ::access(filename.c_str(), R_OK) == 0;
}

// Find the appropriate file, first looking in the local directory, then the
// directory given in settings, and finally the global system directory.
string symbolToFile(symbol *id)
{
  if((string) *id == "-") return "-";
  
  string filename = dirSymbolToFile("", id);
  if(exists(filename)) return filename;

  if(settings::AsyDir) {
    filename = dirSymbolToFile(settings::AsyDir, id);
    if(exists(filename)) return filename;
  }

#ifdef ASYMPTOTE_SYSDIR
  filename = dirSymbolToFile(ASYMPTOTE_SYSDIR, id);
  if(exists(filename)) return filename;
#endif  

  return "";
}
 
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

  // Import plain, if autoplain option is enabled.
  if (settings::autoplain) {
    static absyntax::importdec iplain(position::nullPos(),
				      symbol::trans("plain"));
    iplain.trans(base_coenv);
    me.beginScope(); // NOTE: This is unmatched.
  }
  
  if(!settings::ignoreGUI) {
    string GUIname=buildname(settings::outname,"gui");
    std::ifstream exists(GUIname.c_str());
    if(exists) {
      if(settings::clearGUI) unlink(GUIname.c_str());
      else {
	absyntax::importdec igui(position::nullPos(),
				 symbol::trans(GUIname.c_str()));
	igui.trans(base_coenv);
	me.beginScope();
      }
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
record *genv::loadModule(symbol *id)
{
  // Get the abstract syntax tree.
  absyntax::file *ast = parseModule(id);
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

// Opens and parses the file returning the abstract syntax tree.  If
// there is an unrecoverable parse error, returns null.
absyntax::file *genv::parseModule(symbol *id)
{
  std::string filename = symbolToFile(id);

  if (filename == "")
    return 0;

  // For debugging the lexer and parser that were machine generated.
  yy_flex_debug = 0;
  yydebug = 0;

  if (!setlexer(filename))
    return 0;

  if (yyparse() == 0) return absyntax::root;
  return 0;
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
