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
#include "settings.h"
#include "vm.h"
#include "program.h"
#include "interact.h"
#include "parser.h"
#include "fileio.h"

#include "stack.h"
#include "runtime.h"

using namespace settings;
using std::list;

using absyntax::file;
using trans::genv;
using trans::coenv;
using trans::env;
using trans::coder;
using types::record;

errorstream *em;
using interact::interactive;
using interact::virtualEOF;

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

bool status=true;

namespace loop {

void init()
{
  ShipoutNumber=0;
  if (!em)
    em = new errorstream();
}

void purge()
{
  em->clear();
  outname="";

#ifdef USEGC
  GC_gcollect();
#endif
}

void doPrint(genv& ge, record *m)
{
  // NOTE: Should make it possible to show more code.
  print(cout, m->getInit()->code);
}

// Run (an already translated) module of the given filename.
void doRun(genv& ge, std::string filename)
{
  setPath(startPath());

  vm::stack s;
  s.setInitMap(ge.getInitMap());
  s.load(filename);
  run::exitFunction(&s);
}

void body(string filename)
{
  string basename = stripext(filename,suffix);
  try {
    if (verbose) cout << "Processing " << basename << endl;
    
    if(outname.empty())
      outname=(filename == "-") ? "out" : stripDir(basename);

    if (parseonly) {
      absyntax::file *tree = parser::parseFile(filename);
      em->sync();

      if (!em->errors())
        tree->prettyprint(cout, 0);
      else status=false;
    } else {
      genv ge;
      record *m = ge.getModule(symbol::trans(basename),filename);
      if (!em->errors()) {
	if (translate)
	  doPrint(ge,m);
	else
	  doRun(ge,filename);
      } else status=false;
    }
  } catch (std::bad_alloc&) {
    cerr << "error: out of memory" << endl;
    status=false;
  } catch (handled_error) {
    status=false;
  }
}

void doBatch()
{
  for(int ind=0; ind < numArgs() ; ind++) {
    init();
    body(getArg(ind));
    purge();
  }
}

typedef vm::interactiveStack istack;
using absyntax::runnable;
using absyntax::block;

mem::list<coenv*> estack;
mem::list<vm::interactiveStack*> sstack;

// Abstract base class for the core object being run in line-at-a-time mode, it
// may be a runnable, block, file, or interactive prompt.
struct icore {
  virtual ~icore() {}
  
  virtual void run(coenv &e, istack &s) = 0;
  
  // Wrapper for run used to execute eval within the current environment.
  void wrapper(coenv &e, istack &s) {
    estack.push_back(&e);
    sstack.push_back(&s);
    run(e,s);
    estack.pop_back();
    sstack.pop_back();
  }
  
  void embedded() {
    assert(estack.size() && sstack.size());
    run(*(estack.back()),*(sstack.back()));
  };
};

struct irunnable : public icore {
  runnable *r;

  irunnable(runnable *r)
    : r(r) {}

  void run(coenv &e, istack &s) {
    e.e.beginScope();
    lambda *codelet=r->transAsCodelet(e);
    em->sync();
    if (!em->errors()) {
      s.run(codelet);
    } else {
      e.e.endScope(); // Remove any changes to the environment.
      status=false;
      em->clear();
    }
  }
};

struct itree : public icore {
  absyntax::block *ast;

  itree(absyntax::block *ast)
    : ast(ast) {}

  void run(coenv &e, istack &s) {
    for(list<runnable *>::iterator r=ast->stms.begin(); r != ast->stms.end();
	++r)
      irunnable(*r).wrapper(e, s);
  }
};

struct iprompt : public icore {
  void run(coenv &e, istack &s) {
    while (virtualEOF) {
      virtualEOF=false;
      try {
        file *ast = parser::parseInteractive();
        assert(ast);
        itree(ast).wrapper(e, s);
      } catch (interrupted&) {
        if(em) em->Interrupt(false);
        cout << endl;
      } catch (...) {
        status=false;
      }
    }
  }
};

void doICore(icore &i, bool embedded=false) {
  try {
    assert(em && !em->errors());
    if(embedded) i.embedded();
    else {
      genv ge;
      env base_env(ge);
      coder base_coder;
      coenv e(base_coder,base_env);

      vm::interactiveStack s;
      s.setInitMap(ge.getInitMap());

      if (settings::autoplain)
	irunnable(absyntax::autoplainRunnable()).wrapper(e, s);

      // Now that everything is set up, run the core.
      i.wrapper(e, s);

      if(settings::listvariables)
	base_env.list();
    
      run::exitFunction(&s);
      em->clear();
    }
  } catch (std::bad_alloc&) {
    cerr << "error: out of memory" << endl;
    status=false;
  } catch(handled_error) {
    status=false;
  }
}
      
void doIRunnable(runnable *r, bool embedded=false) {
  assert(r);
  irunnable i(r);
  doICore(i,embedded);
}

void doITree(block *tree, bool embedded=false) {
  assert(tree);
  itree i(tree);
  doICore(i,embedded);
}

void doIFile(const string& filename) {
  init();

  string basename = stripext(filename,suffix);
  if (verbose) cout << "Processing " << basename << endl;
  if(outname.empty())
    outname=stripDir(basename);

  doITree((filename == "" ? parser::parseString 
	   : parser::parseFile)(filename));

  purge();
}

void doIPrompt() {
  init();
  outname="out";
  
  iprompt i;
  doICore(i);

  purge();
}

} // namespace loop

int main(int argc, char *argv[])
{
#ifdef USEGC
  GC_free_space_divisor = 2;
  GC_dont_expand = 0;
  GC_INIT();
#endif  

  setOptions(argc,argv);

  fpu_trap(trap);
  setsignal(signalHandler);
  if(interactive) signal(SIGINT,interruptHandler);

  cout.precision(DBL_DIG);

  try {
    if (interactive)
      loop::doIPrompt();
    else
      if(numArgs() == 0) {
	loop::doIFile("");
      } else for(int ind=0; ind < numArgs() ; ind++)
	loop::doIFile(string(getArg(ind)));
  } catch (...) {
    cerr << "error: exception thrown.\n";
    status=false;
  }
  return status ? 0 : 1;
}
