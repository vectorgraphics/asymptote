#include <iostream>
#include <cfloat>
#include <csignal>
#include <cstdlib>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifdef HAVE_LIBSIGSEGV
#include <sigsegv.h>
#endif

#include "types.h"
#include "errormsg.h"
#include "fpu.h"
#include "genv.h"
#include "stm.h"
#include "camp.tab.h"
#include "settings.h"
#include "stack.h"
#include "interact.h"
#include "parser.h"

using namespace settings;
using namespace std;

using absyntax::file;
using trans::genv;
using types::record;

errorstream *em;
using interact::interactive;
using interact::virtualEOF;
using interact::rejectline;

#ifdef HAVE_LIBSIGSEGV
void stackoverflow_handler (int, stackoverflow_context_t)
{
  em->runtime(vm::getPos());
  cout << "Stack overflow" << endl;
  abort();
}

int sigsegv_handler (void *, int emergency)
{
  if(!emergency) return 0; // Really a stack overflow
  em->runtime(vm::getPos());
  cout << "Segmentation fault" << endl;
  cout << "Please report this programming error to" << endl 
       << BUGREPORT << endl;
  abort();
}
#endif 

void setsignal(RETSIGTYPE (*handler)(int))
{
#ifdef HAVE_LIBSIGSEGV
  char mystack[16384];
  if(stackoverflow_install_handler(&stackoverflow_handler,
				   mystack,sizeof (mystack)) < 0) exit(1);
  if(sigsegv_install_handler (&sigsegv_handler) < 0) exit (1);
#endif
  signal(SIGBUS,handler);
  signal(SIGFPE,handler);
}

void signalHandler(int)
{
  if(em) em->runtime(vm::getPos());
  signal(SIGBUS,SIG_DFL);
  signal(SIGFPE,SIG_DFL);
}

void interruptHandler(int)
{
  if(em) em->Interrupt(true);
}

int main(int argc, char *argv[])
{
  setOptions(argc,argv);

  fpu_trap(trap);
  setsignal(signalHandler);
  if(interactive) signal(SIGINT,interruptHandler);

  std::cout.precision(DBL_DIG);
  int status = 0;
  
  for(int ind=0; ind < numArgs() || (interactive && virtualEOF); ind++) {
    virtualEOF=false;
    ShipoutNumber=0;
    
    string module_name = interactive ? "<stdin>" : getArg(ind);
    size_t p=findextension(module_name,suffix);
    if (p < string::npos) module_name.erase(p);
    
    try {
      outnameStack=new std::list<std::string>;
      
      if (verbose) cout << "Processing " << module_name << endl;
    
      if(outname.empty()) 
	outname=(module_name == "<stdin>") ? "out" : module_name;
    
      symbol *id = symbol::trans(module_name);
    
      em = new errorstream();

      genv ge;

      // Import plain, if autoplain option is enabled.
      if (autoplain)
        ge.loadPlain();
      if (!ignoreGUI)
        ge.loadGUI(outname);

      absyntax::file *tree = interact::interactive ?
        parser::parseInteractive() : parser::parseFile(module_name);
      if (parseonly) {
        em->sync();
        if (!em->errors() && (tree != 0))
          tree->prettyprint(std::cout, 0);
      } else {
        record *m = ge.loadModule(id,tree);
        if (m) {
          lambda *l = ge.bootupModule(m);
          assert(l);
        
          if (em->errors() == false) {
            if (translate) {
              // NOTE: Should make it possible to show more code.
              print(cout, l->code);
              cout << "\n";
              print(cout, m->getInit()->code);
            } else {
              vm::stack s;
	      setPath(startPath());
              s.run(l);
            }
          }
        } else {
          if (em->errors() == false)
            cerr << "error: could not load module '" << *id << "'" << endl;
        }
      }
    } catch (std::bad_alloc&) {
      cerr << "error: out of memory" << endl;
      ++status;
    } catch (handled_error) {
      ++status;
    } catch (interrupted) {
      if(em) em->Interrupt(false);
      cerr << endl;
      run::cleanup(true);
    } catch (const char* s) {
      cerr << "error: " << s << endl;
      ++status;
    } catch (...) {
      cerr << "error: exception thrown processing '" << module_name << "'\n";
      ++status;
    }
    
    rejectline=em->warnings();
    if(rejectline) virtualEOF=true;
    
    delete em; em = 0;
    delete outnameStack; outnameStack = 0;
    outname="";
    memory::free();
  }
  return status;
}
