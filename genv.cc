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

using namespace std;
using namespace types;
using vm::inst;

// The lexical analysis and parsing functions used by parseFile.
extern bool setlexer(std::string filename);
extern bool yyparse(void);
extern int yydebug;
extern int yy_flex_debug;

namespace trans {

namespace {

// Adds the appropriate directory and suffix to the name.
// Note that "asy examples/blah.asy" works even in the examples/ directory,
// if examples is in ASYMPTOTE_DIR.
string dirSymbolToFile(string s, symbol *id)
{
  ostringstream buf;
  if (!s.empty())
    buf << s << "/";
  
  if(((string)*id).find('.') != string::npos) 
    buf << *id;
  else
    buf << *id << "." << settings::suffix;
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

  filename = dirSymbolToFile(settings::getAsyDir(), id);
  if(exists(filename)) return filename;

#ifdef ASYMPTOTE_SYSDIR
  filename = dirSymbolToFile(ASYMPTOTE_SYSDIR, id);
  if(exists(filename)) return filename;
#endif  

  return "";
}
 
} // private namespace

genv::genv()
 : dummy_env(*this)
{
  types::initializeCasts();
  types::initializeInitializers();
  base_tenv(te);
  base_venv(ve);
  base_menv(me);
  // Import plain, if that option is enabled.
  if (settings::autoplain) {
    static as::importdec iplain(position::nullPos(),symbol::trans("plain"));
    iplain.trans(dummy_env);
    me.beginScope();
  }
  
  if(!settings::ignoreGUI) {
    string GUIname=buildname(settings::outname,"gui");
    std::ifstream exists(GUIname.c_str());
    if(exists) {
      if(settings::clearGUI) unlink(GUIname.c_str());
      else {
	as::importdec igui(position::nullPos(),symbol::trans(GUIname.c_str()));
	igui.trans(dummy_env);
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
  as::file *ast = parseModule(id);
  em->sync();

  if (!ast)
    return 0;

  //ast->prettyprint(stdout, 0);
 
  // Create the new module.
  record *r = dummy_env.newRecord(id);

  // Add it to the table of modules.
  modules.enter(id, r);

  // Create an environment to translate the module.
  // File-level modules have static fields by default.
  env e(r, dummy_env, DEFAULT_STATIC);
  //env &e = new env(r, dummy_env, DEFAULT_DYNAMIC);

  // Translate the abstract syntax.
  ast->transAsRecordBody(e, r);
  em->sync();

  return r;
}

// Opens and parses the file returning the abstract syntax tree.  If
// there is an unrecoverable parse error, returns null.
as::file *genv::parseModule(symbol *id)
{
  std::string filename = symbolToFile(id);

  if (filename == "")
    return 0;

  // For debugging the lexer and parser that were machine generated.
  yy_flex_debug = 0;
  yydebug = 0;

  if (!setlexer(filename))
    return 0;

  if (yyparse() == 0) return as::root;
  return 0;
}

// Returns a function that statically initializes all loaded modules.
// Then runs the dynamic initializer of r.
// This should be the lowest-level function run by the stack.
lambda *genv::bootupModule(record *r)
{
  // Encode the record dynamic instantiation.
  if (!dummy_env.encode(r->getLevel()->getParent())) {
    em->compiler();
    *em << "invalid bootup record";
    em->sync();
    return 0;
  }


  // Encode the allocation.
  dummy_env.encode(inst::alloc);
  inst i; i.r = r->getRuntime();
  dummy_env.encode(i);
  dummy_env.encode(inst::pop);

  // Return the finished function.
  return dummy_env.close();
}

} // namespace trans
