/*****
 * process.cc
 * Andy Hammerlindl 2006/08/19 
 *
 * Handles processing blocks of code (including files, strings, and the
 * interactive prompt, for listing and parse-only modes as well as actually
 * running it.
 *****/

#include "types.h"
#include "errormsg.h"
#include "genv.h"
#include "stm.h"
#include "settings.h"
#include "vm.h"
#include "program.h"
#include "interact.h"
#include "envcompleter.h"
#include "parser.h"
#include "fileio.h"

#include "stack.h"
#include "runtime.h"
#include "texfile.h"

#include "process.h"

namespace run {
  void cleanup();
  void exitFunction(vm::stack *Stack);
  void updateFunction(vm::stack *Stack);
}

namespace vm {
  bool indebugger;  
}

using namespace settings;
using std::list;

using absyntax::file;
using trans::genv;
using trans::coenv;
using trans::env;
using trans::coder;
using types::record;

using interact::interactive;
using interact::uptodate;

void init()
{
  vm::indebugger=false;
  setPath(startPath());  /* On second and subsequent calls to init, sets the
                            path to what it was when the program started. */
  ShipoutNumber=0;
  if(!em)
    em = new errorstream();
}

void purge()
{
#ifdef USEGC
  GC_gcollect();
#endif
}

using absyntax::runnable;
using absyntax::block;

// How to run a runnable in runnable-at-a-time mode.
bool runRunnable(runnable *r, coenv &e, istack &s) {
  e.e.beginScope();
  lambda *codelet=r->transAsCodelet(e);
  em->sync();
  if(!em->errors()) {
    if(getSetting<bool>("translate")) print(cout,codelet->code);
    s.run(codelet);

    // NOTE: May want to add a "e.e.collapseTopScope()" here to keep scoping
    // from getting out-of-hand.
  } else {
    e.e.endScope(); // Remove any changes to the environment.

    // Should an interactive error hurt the status?
    em->statusError();

    return false;   
  }
  return true;
}

void runAutoplain(coenv &e, istack &s) {
  absyntax::runnable *r=absyntax::autoplainRunnable();
  runRunnable(r,e,s);
}


// Abstract base class for the core object being run in line-at-a-time mode, it
// may be a block of code, file, or interactive prompt.
struct icore {
  virtual ~icore() {}

  virtual void doParse() = 0;
  virtual void doList() = 0;

private:
  // NOTE: Get this out of here!
  std::list<string> TeXpipepreamble_save;
  std::list<string> TeXpreamble_save;
public:

  virtual void preRun(coenv &e, istack &s) {
    TeXpipepreamble_save = std::list<string>(camp::TeXpipepreamble);
    TeXpreamble_save = std::list<string>(camp::TeXpreamble);

    if(getSetting<bool>("autoplain"))
      runAutoplain(e,s);
  }

  virtual void run(coenv &e, istack &s) = 0;

  virtual void postRun(coenv &e, istack &s) {
    // Run the exit function in non-interactive mode.
    // NOTE: resetenv used to do run::cleanup() instead.
    bool temp=interactive;
    interactive=false;
    run::exitFunction(&s);
    interactive=temp;

    camp::TeXpipepreamble=TeXpipepreamble_save;
    camp::TeXpreamble=TeXpreamble_save;
  }

  virtual void doRun() {
    assert(em);
    em->sync();
    if(em->errors())
      return;

    try {
      purge();

      genv ge;
      env base_env(ge);
      coder base_coder;
      coenv e(base_coder,base_env);

      vm::interactiveStack s;
      s.setInitMap(ge.getInitMap());
      s.setEnvironment(&e);

      preRun(e,s);

      // Now that everything is set up, run the core.
      run(e,s);

      postRun(e,s);

    } catch (std::bad_alloc&) {
      cerr << "error: out of memory" << endl;
      em->statusError();
    } catch (handled_error) {
      em->statusError();
      run::cleanup();
    }

    em->clear();
  }

  virtual void process() {
    if (getSetting<bool>("parseonly"))
      doParse();
    else if (getSetting<bool>("listvariables"))
      doList();
    else
      doRun();
  }
};

// Abstract base class for one-time processing of an abstract syntax tree.
class itree : public icore {
  mem::string name;

  block *cachedTree;
public:
  itree(mem::string name="<unnamed>")
    : name(name), cachedTree(0) {}

  // Build the tree, possibly throwing a handled_error if it cannot be built.
  virtual block *buildTree() = 0;

  virtual block *getTree() {
    if (cachedTree==0) {
      try {
        cachedTree=buildTree();
      } catch (handled_error) {
        em->statusError();
        return 0;
      }
    }
    return cachedTree;
  }

  virtual mem::string getName() {
    return name;
  }

  void doParse() {
    block *tree=getTree();
    em->sync();
    if(tree && !em->errors())
      tree->prettyprint(cout, 0);
  }

  void doList() {
    block *tree=getTree();
    if (tree) {
      genv ge;
      record *r=tree->transAsFile(ge, symbol::trans(getName()));
      r->e.list(r);
    }
  }

  void run(coenv &e, istack &s) {
    block *tree=getTree();
    if (tree) {
      for(list<runnable *>::iterator r=tree->stms.begin();
          r != tree->stms.end(); ++r)
        if(!em->errors() || getSetting<bool>("debug"))
          runRunnable(*r,e,s);
    }
  }

  void doRun() {
    // Don't prepare an environment to run the code if there isn't any code.
    if (getTree()) {
      icore::doRun();
    }
  }
};

class icode : public itree {
  block *tree;

public:
  icode(block *tree, mem::string name="<unnamed>")
    : itree(name), tree(tree) {}

  block *buildTree() {
    return tree;
  }
};

class istring : public itree {
  mem::string str;

public:
  istring(mem::string str, mem::string name="<eval>")
    : itree(name), str(str) {}

  block *buildTree() {
    return parser::parseString(str, getName());
  }
};

class ifile : public itree {
  mem::string filename;
  mem::string basename;
  mem::string outname_save;

public:
  ifile(mem::string filename)
    : itree(filename),
      filename(filename), basename(stripext(filename, suffix)) {}

  block *buildTree() {
    return filename!="" ? parser::parseFile(filename) : 0;
  }

  // Should fix stripDir to take mem::strings.
  static mem::string stripDirHack(mem::string ms) {
    std::string s=ms;
    stripDir(s);
    return (mem::string)s;
  }

  void preRun(coenv& e, istack& s) {
    outname_save=getSetting<mem::string>("outname");
    if(outname_save.empty())
      Setting("outname")=
        (mem::string)((filename == "-") ? "out" : stripDirHack(basename));

    itree::preRun(e, s);
  }

  void postRun(coenv &e, istack& s) {
    itree::postRun(e, s);

    Setting("outname")=(mem::string)outname_save;
  }

  void process() {
    init();

    if (verbose >= 1)
      cout << "Processing " << basename << endl;
    
    try {
      itree::process();
    }
    catch (handled_error) {
      em->statusError();
    }
  }
};

void printGreeting() {
  if(!getSetting<bool>("quiet"))
    cout << "Welcome to " << PROGRAM << " version " << VERSION
	 << " (to view the manual, type help)" << endl;
}

// Add a semi-colon terminator, if one is not there.
mem::string terminateLine(const mem::string line) {
  return (!line.empty() && *(line.rbegin())!=';') ? (mem::string)(line+";") :
                                                    line;
}

class iprompt : public icore {
  // Use mem::string throughout.
  typedef mem::string string;

  // Flag that is set to false to signal the prompt to exit.
  bool running;

  // Flag that is set to restart the main loop once it has exited.
  bool restart;

  // Code ran at start-up.
  string startline;
  //block *startcode;

  // Commands are chopped into the starting word and the rest of the line.
  struct commandLine {
    string line;
    string word;
    string rest;
    
    commandLine(string line) : line(line) {
      string::iterator c=line.begin();

      // Skip leading whitespace
      while (c != line.end() && isspace(*c))
        ++c;

      // Only handle identifiers starting with a letter.
      if (c != line.end() && isalpha(*c)) {
        // Store the command name.
        while (c != line.end() && (isalnum(*c) || *c=='_')) {
          word.push_back(*c);
          ++c;
        }
      }

      // Copy the rest to rest.
      while (c != line.end()) {
        rest.push_back(*c);
        ++c;
      }

#if 0
      cerr << "line: " << line << endl;
      cerr << "word: " << word << endl;
      cerr << "rest: " << rest << endl;
      cerr << "simple: " << simple() << endl;
#endif
    }

    // Simple commands have at most spaces or semicolons after the command word.
    bool simple() {
      for (string::iterator c=rest.begin(); c != rest.end(); ++c)
        if (!isspace(*c) && *c != ';')
          return false;
      return true;
    }
  };


  // The interactive prompt has special functions which cannot be implemented as
  // normal functions.  These special funtions take a commandLine as an argument
  // and return true if they can handled the command.  If false is return, the
  // line is treated as a normal line of code.
  // commands is a map of command names to methods which implement the commands.
  typedef bool (iprompt::*command)(commandLine);
  typedef mem::map<CONST string, command> commandMap;
  commandMap commands;

  bool quit(commandLine cl) {
    if (cl.simple()) {
      // Don't store quit commands in the history file.
      interact::deleteLastLine();

      running=false;
      return true;
    }
    else return false;
  }

  bool reset(commandLine cl) {
    if (cl.simple()) {
      running=false;
      restart=true;
      //startcode=0;
      startline="";

      uptodate=true;
      purge();

      return true;
    }
    else return false;
  }

  bool help(commandLine cl) {
    if (cl.simple()) {
      popupHelp();
      return true;
    }
    else return false;
  }


  bool input(commandLine cl) {
    mem::string prefix="erase(); include ";
    mem::string line=terminateLine(prefix+cl.rest);
    //block *code=parser::parseString(line, "<input>");
    //if (!em->errors() && code) {
    if (true) {
      running=false;
      restart=true;
      //startcode=code;
      startline=line;
    } else {
      em->sync();
      em->clear();
    }
    return true;
  }

  void initCommands() {
#define ADDCOMMAND(name, func) \
    commands[#name]=&iprompt::func

    // keywords.pl looks for ADDCOMMAND to identify special commands in the
    // auto-completion.
    ADDCOMMAND(quit,quit);
    ADDCOMMAND(q,quit);
    ADDCOMMAND(exit,quit);
    ADDCOMMAND(reset,reset);
    ADDCOMMAND(help, help);
    ADDCOMMAND(input, input);

#undef ADDCOMMAND
  }

  bool handleCommand(string line) {
    commandLine cl(line);
    if (cl.word != "") {
      commandMap::iterator p=commands.find(cl.word);
      if (p != commands.end()) {
        // Execute the command.
        command &com=p->second;
        return (this->*com)(cl);
      }
      else
        return false;
    }
    else
      return false;
  }

  void runLine(coenv &e, istack &s, string line) {
    try {
      istring i(terminateLine(line), "-");
      i.run(e,s);

      if(!uptodate)
        run::updateFunction(&s);

    } catch (handled_error) {
      vm::indebugger=false;
    } catch (interrupted&) {
      // Turn off the interrupted flag.
      if (em)
        em->Interrupt(false);
      cout << endl;
    }

    // Ignore errors from this line when trying to run subsequent lines.
    em->clear();

    purge(); // Close any files that have gone out of scope.
  }

  void runStartCode(coenv &e, istack &s) {
#if 0
    if (startcode) {
      try {
        icode(startcode).run(e,s);
      } catch (handled_error) {}
    }
#endif
    if (!startline.empty())
      runLine(e, s, startline);
  }

public:
  //iprompt() : running(false), restart(false), startcode(0) {
  iprompt() : running(false), restart(false), startline("") {
    initCommands();
  }

  void doParse() {
    cerr << "iprompt::doParse() not yet implemented" << endl;
  }

  void doList() {
    cerr << "iprompt::doParse() not yet implemented" << endl;
  }

  void run(coenv &e, istack &s) {
    running=true;
    interact::setCompleter(new trans::envCompleter(e.e));

    runStartCode(e, s);

    while (running) {
      // Read a line from the prompt.
      string line=interact::simpleline();

      // Copied from the old interactive prompt.  Needs to be moved.
      ShipoutNumber=0;

      // Check if it is a special command.
      if (handleCommand(line))
        continue;
      else
        runLine(e, s, line);
    }
  }

  void process() {
    printGreeting();
    interact::init_interactive();

    do {
      try {
	init();
	restart=false;
	icore::process();
      } catch(interrupted&) {
	if(em) em->Interrupt(false);
      }
    } while(restart);
      
    interact::cleanup_interactive();
  }
};

void processCode(absyntax::block *code) {
  icode(code).process();
}
void processString(mem::string string) {
  istring(string).process();
}
void processFile(mem::string filename) {
  ifile(filename).process();
}
void processPrompt() {
  iprompt().process();
}

void runCode(absyntax::block *code) {
  icode(code).doRun();
}
void runString(mem::string string) {
  istring(string).doRun();
}
void runFile(mem::string filename) {
  ifile(filename).doRun();
}
void runPrompt() {
  iprompt().doRun();
}

void runCodeEmbedded(absyntax::block *code, trans::coenv &e, istack &s) {
  icode(code).run(e,s);
}
void runStringEmbedded(mem::string string, trans::coenv &e, istack &s) {
  istring(string).run(e,s);
}
void runFileEmbedded(mem::string filename, trans::coenv &e, istack &s) {
  ifile(filename).run(e,s);
}
void runPromptEmbedded(trans::coenv &e, istack &s) {
  iprompt().run(e,s);
}

void doUnrestrictedList() {
  genv ge;
  env base_env(ge);
  coder base_coder;
  coenv e(base_coder,base_env);

  if (getSetting<bool>("autoplain"))
    absyntax::autoplainRunnable()->trans(e);

  e.e.list(0);
}

