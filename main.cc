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

int status = 0;

namespace loop {

void init()
{
  if (interactive) virtualEOF=false;
  ShipoutNumber=0;

  outnameStack=new std::list<std::string>;

  em = new errorstream();
}


void cleanup()
{
  if (interactive) {
    rejectline=em->warnings();
    if(rejectline) virtualEOF=true;
  }
  
  delete em; em = 0;
  delete outnameStack; outnameStack = 0;
  outname="";
  memory::free();
}

void body(string filename) // TODO: Refactor
{
  string basename = stripext(filename,suffix);
  try {
    init();
  
    if (verbose) cout << "Processing " << basename << endl;
    
    if(outname.empty()) 
      outname=(filename == "-") ? "out" : basename;

    genv ge;
    
    ge.autoloads(outname);
    
    absyntax::file *tree = interactive ?
      parser::parseInteractive() : parser::parseFile(filename);
    em->sync();

    if (parseonly) {
      if (!em->errors())
        tree->prettyprint(std::cout, 0);
    } else {
      record *m = ge.loadModule(symbol::trans(basename),tree);
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
            setPath(startPath());
            vm::run(l);
          }
        }
      } else {
        if (em->errors() == false)
          cerr << "error: could not load module '" << basename << "'" << endl;
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
    cerr << "error: exception thrown processing '" << basename << "'\n";
    ++status;
  }

  cleanup();
}

}


int main(int argc, char *argv[])
{
  setOptions(argc,argv);

  fpu_trap(trap);
  setsignal(signalHandler);
  if(interactive) signal(SIGINT,interruptHandler);

  std::cout.precision(DBL_DIG);

  if (interactive) {
    while (virtualEOF)
      loop::body("-"); 
  } else {
    for(int ind=0; ind < numArgs() ; ind++)
      loop::body(getArg(ind));
  }
  return status;
}
