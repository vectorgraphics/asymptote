#include <iostream>
#include <cfloat>
#include <csignal>

#include "types.h"
#include "errormsg.h"
#include "fpu.h"
#include "genv.h"
#include "stm.h"
#include "camp.tab.h"
#include "settings.h"
#include "stack.h"
#include "interact.h"
#include "texfile.h"
#include "pipestream.h"
#include "picture.h"

using namespace settings;
using namespace std;

using as::file;
using trans::genv;
using types::record;

errorstream *em;
using interact::interactive;
using interact::virtualEOF;
using interact::rejectline;

void setsignal(void (*handler)(int))
{
  signal(SIGBUS,handler);
  signal(SIGFPE,handler);
  signal(SIGSEGV,handler);
}

void signalHandler(int)
{
  em->runtime(lastpos);
  setsignal(SIG_DFL);
}

int main(int argc, char *argv[])
{
  setOptions(argc,argv);

  fpu_trap(trap);
  setsignal(signalHandler);
  signal(SIGCHLD, SIG_IGN); // Flush exited child processes (avoid zombies)

  if(settings::numArgs() == 0) {
    interactive=true;
    view=1;
    cout << "Welcome to " << settings::PROGRAM << " version " 
	 << settings::VERSION << " (interactive mode)" << endl;
  }
  
  std::cout.precision(DBL_DIG);
  
  int status = 0;
  
  for(int ind=0; ind < settings::numArgs() || (interactive && virtualEOF);
      ind++) {
    settings::reset();
    virtualEOF=false;
    ShipoutNumber=0;
    
    string module_name = interactive ? "-" : getArg(ind);
    size_t p=module_name.rfind("."+settings::suffix);
    if (p < string::npos) module_name.erase(p);
    
    try {
      outnameStack=new std::list<std::string>;
      
      if (verbose) cout << "Processing " << module_name << endl;
    
      if(outname == "") outname=(module_name == "-") ? "out" : module_name;
    
      symbol *id = symbol::trans(module_name);
    
      em = new errorstream();

      genv ge;

      if (parseonly) {
        as::file *tree = ge.parseModule(id);
        em->sync();
        if (!em->errors())
          tree->prettyprint(std::cout, 0);
      } else {
        record *m = ge.loadModule(id);
        if (m) {
          lambda *l = ge.bootupModule(m);
          assert(l);
        
          if (em->errors() == false) {
            if (translate) {
              // NOTE: Should make it possible to show more code.
              print(cout, l->code);
              cout << "\n";
              print(cout, m->getRuntime()->init->code);
            } else {
              vm::stack s(0);
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
    } catch (const char* s) {
      cerr << "error: " << s << endl;
      ++status;
    } catch (...) {
      cerr << "error: exception thrown processing '" << module_name << "'\n";
      ++status;
    }

    rejectline=em->errors();
    if(rejectline) virtualEOF=true;
    delete em; em = 0;
    delete outnameStack; outnameStack = 0;
    if(camp::TeXcontaminated) {
      camp::TeXpreamble.clear();
      tex.pipeclose();
      TeXinitialized=false;
    }
    outname="";
    mempool::free();
  }
  return status;
}
