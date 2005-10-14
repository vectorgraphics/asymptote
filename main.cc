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

using namespace settings;
using std::list;

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
  ShipoutNumber=0;
  outnameStack=new list<string>;
  em = new errorstream();
}

void purge()
{
  delete em; em = 0;
  delete outnameStack; outnameStack = 0;
  outname="";
//  camp::file::free();
#ifdef USEGC
  GC_gcollect();
#endif
}

void doPrint(genv& ge, record *m)
{
  // NOTE: Should make it possible to show more code.
  print(cout, m->getInit()->code);
}

// Run (an already translated)
void doRun(genv& ge, std::string filename)
{
  setPath(startPath()); // ???

  vm::stack s;
  s.setInitMap(ge.getInitMap());
  s.load(filename);
}

void body(string filename) // TODO: Refactor
{
  string basename = stripext(filename,suffix);
  try {
    if (verbose) cout << "Processing " << basename << endl;
    
    if(outname.empty())
      outname=(filename == "-") ? "out" : basename;

    if (parseonly) {
      absyntax::file *tree = parser::parseFile(filename);
      em->sync();

      if (!em->errors())
        tree->prettyprint(cout, 0);
    } else {
      genv ge;
#if 0 
      ge.autoloads(outname);
      if(settings::listonly) {
        ge.list();
        if(filename == "") return;
      }
#endif

      record *m = ge.getModule(symbol::trans(basename),filename);
      if (!em->errors()) {
	if (translate)
	  doPrint(ge,m);
	else
	  doRun(ge,filename);
      }
    }
  } catch (std::bad_alloc&) {
    cerr << "error: out of memory" << endl;
    ++status;
  } catch (handled_error) {
    ++status;
  }
}

void doInteractive()
{
  while (virtualEOF) {
    virtualEOF=false;
    init();
    try {
      body("-");
    } catch (interrupted&) {
      if(em) em->Interrupt(false);
      cerr << endl;
      run::cleanup();
    }
    rejectline=em->warnings();
    if(rejectline) {
      virtualEOF=true;
      run::cleanup();
    }
    purge();
  }
}

void doBatch()
{
#if 0
  if(listonly && numArgs() == 0) {
    init();
    body("");
    purge();
  } else
#endif

  for(int ind=0; ind < numArgs() ; ind++) {
    init();
    body(getArg(ind));
    purge();
  }
}

#if 0
typedef vm::interactiveStack istack;
using absyntax::runnable;
using absyntax::file;

void doIRunnable(absyntax::runnable *r, genv &ge, istack &s) {
  lambda *codelet=ge.trans(r);
  print(cout, codelet->code);
  cout << "\n";
  s.run(codelet);
}

void doITree(file *ast, genv &ge, istack &s) {
  for (list<runnable *>::iterator r=ast->stms.begin();
       r!=ast->stms.end();
       ++r)
    doIRunnable(*r, ge, s);
}

void doIFile(const char *name,  genv &ge, istack &s) {
  cout << "ifile: " << name << endl;
  file *ast=parser::parseFile(name);
  assert(ast);
  doITree(ast, ge, s);
}

void doIBatch()
{
  cout << "ibatch\n";
  if(listonly && numArgs() == 0) {
    init();
    body("");
    purge();
  }
  else {
    init();

    genv ge;
    vm::interactiveStack s;

    for(int ind=0; ind < numArgs() ; ind++)
      doIFile(getArg(ind), ge, s);

    purge();
  }
}
#endif

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
      loop::doInteractive();
    else if (laat) {
      cout << "laat = " << laat << endl;
      cout << "not working right now\n";
      //loop::doIBatch();
    }
    else
      loop::doBatch();
  } catch (...) {
    cerr << "error: exception thrown.\n";
    ++status;
  }
  return status;
}
