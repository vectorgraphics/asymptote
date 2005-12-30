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
#include "runtime.h"
#include "parser.h"
#include "locate.h"
#include "interact.h"

using namespace types;
using settings::getSetting;
using settings::Setting;

namespace trans {

genv::genv()
  : imap()
{
  // Add settings as a module.  This is so that the init file ~/.asy/config.asy
  // can set settings.
  imap["settings"]=settings::getSettingsModule();

  // Translate plain in advance, if we're using autoplain.
  if(getSetting<bool>("autoplain")) {
    Setting("autoplain")=false;

    // Translate plain without autoplain.
    getModule(symbol::trans("plain"), "plain");

    Setting("autoplain")=true;
  }
}

record *genv::loadModule(symbol *id, mem::string filename) {
  if(settings::verbose > 1)
    std::cerr << "Loading " <<  filename << std::endl;
    
  // Get the abstract syntax tree.
  absyntax::file *ast = parser::parseFile(filename);
  
  inTranslation.push_front(filename);

  em->sync();
  
  // Create the new module.
  record *r = new record(id, new frame(0,0));

  // Create coder and environment to translate the module.
  // File-level modules have dynamic fields by default.
  coder c(r, 0);
  env e(*this);
  coenv ce(c, e);

  // Translate the abstract syntax.
  ast->transAsFile(ce, r);
  em->sync();
  
  inTranslation.remove(filename);

  return r;
}

void genv::checkRecursion(mem::string filename) {
  if (find(inTranslation.begin(), inTranslation.end(), filename) !=
         inTranslation.end()) {
    em->sync();
    *em << "error: recursive loading of module '" << filename << "'\n";
    em->sync();
    throw handled_error();
  }
}

record *genv::getModule(symbol *id, mem::string filename) {
  checkRecursion(filename);

  record *r=imap[filename];
  if (r)
    return r;
  else {
    record *r=loadModule(id, filename);
    // Don't add an erroneous module to the dictionary in interactive mode, as
    // the user may try to load it again.
    if (!interact::interactive || !em->errors())
      imap[filename]=r;

    return r;
  }

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
