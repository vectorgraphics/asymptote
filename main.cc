#include <iostream>
#include <csignal>
#include <cstdlib>
#include <cerrno>
#include <sys/wait.h>
#include <sys/types.h>

#include "common.h"

#ifdef HAVE_LIBSIGSEGV
#include <sigsegv.h>
#endif

#include "errormsg.h"
#include "fpu.h"
#include "settings.h"
#include "locate.h"
#include "interact.h"
#include "process.h"

#include "stack.h"

using namespace settings;

errorstream em;
using interact::interactive;

namespace run {
  void purge();
}
  
#ifdef HAVE_LIBSIGSEGV
void stackoverflow_handler (int, stackoverflow_context_t)
{
  em.runtime(vm::getPos());
  cerr << "Stack overflow" << endl;
  abort();
}

int sigsegv_handler (void *, int emergency)
{
  if(!emergency) return 0; // Really a stack overflow
  em.runtime(vm::getPos());
  cerr << "Segmentation fault" << endl;
  cerr << "Please report this programming error to" << endl 
       << BUGREPORT << endl;
  abort();
}
#endif 

void setsignal(RETSIGTYPE (*handler)(int))
{
#ifdef HAVE_LIBSIGSEGV
  char mystack[16384];
  stackoverflow_install_handler(&stackoverflow_handler,
				mystack,sizeof (mystack));
  sigsegv_install_handler(&sigsegv_handler);
#endif
  signal(SIGBUS,handler);
  signal(SIGFPE,handler);
}

void signalHandler(int)
{
  // Print the position and trust the shell to print an error message.
  em.runtime(vm::getPos());

  signal(SIGBUS,SIG_DFL);
  signal(SIGFPE,SIG_DFL);
}

void interruptHandler(int)
{
  em.Interrupt(true);
}

// Run the config file.
void doConfig(string filename) {
  if(!filename.empty()) {
    string file = locateFile(filename);
    if(!file.empty()) {
      bool autoplain=getSetting<bool>("autoplain");
      bool listvariables=getSetting<bool>("listvariables");
      if(autoplain) Setting("autoplain")=false; // Turn off for speed.
      if(listvariables) Setting("listvariables")=false;

      runFile(file);

      if(autoplain) Setting("autoplain")=true;
      if(listvariables) Setting("listvariables")=true;
    }
  }
}

int main(int argc, char *argv[])
{
  setsignal(signalHandler);

  try {
    setOptions(argc,argv);
  } catch(handled_error) {
    em.statusError();
  }
  
  fpu_trap(trap());

  if(interactive) {
    signal(SIGINT,interruptHandler);
    processPrompt();
  } else if (getSetting<bool>("listvariables") && numArgs()==0) {
    try {
      doUnrestrictedList();
    } catch(handled_error) {
      em.statusError();
    } 
  } else {
    int n=numArgs();
    if(n == 0) 
      processFile("-");
    else
    for(int ind=0; ind < n; ind++) {
      processFile(string(getArg(ind)),n > 1);
      try {
        if(ind < n-1)
          setOptions(argc,argv);
      } catch(handled_error) {
        em.statusError();
      } 
    }
  }

  if(getSetting<bool>("wait")) {
    int status;
    while(wait(&status) > 0);
  }
  
  return em.processStatus() ? 0 : 1;
}
