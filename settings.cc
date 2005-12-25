/*****
 * settings.cc
 * Andy Hammerlindl 2004/05/10
 *
 * Declares a list of global variables that act as settings in the system.
 *****/

#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cerrno>
#include <vector>
#include <sys/stat.h>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#if HAVE_GNU_GETOPT_H
#include <getopt.h>
#else
#include "getopt.h"
#endif

#include "util.h"
#include "settings.h"
#include "pen.h"
#include "interact.h"
#include "locate.h"
#include "lexical.h"

#include "memory.h"
#include "record.h"
#include "env.h"
#include "item.h"
#include "refaccess.h"

#ifdef MSDOS
const bool msdos=true;
#else
const bool msdos=false;
#endif

using std::vector;
using vm::item;

using trans::itemRefAccess;
using trans::refAccess;
using trans::varEntry;


namespace settings {
  
const char PROGRAM[]=PACKAGE_NAME;
const char VERSION[]=PACKAGE_VERSION;
const char BUGREPORT[]=PACKAGE_BUGREPORT;

// The name of the program (as called).  Used when displaying help info.
char *argv0;

typedef ::option c_option;

types::dummyRecord settingsModule(symbol::trans("settings"));

types::record *getSettingsModule() {
  return &settingsModule;
}

// The dictionaries of long options and short options.
class option;
typedef mem::map<const mem::string, option *> optionsMap_t;
optionsMap_t optionsMap;
typedef mem::map<const char, option *> codeMap_t;
codeMap_t codeMap;
  
struct option : public gc {
  mem::string name;
  char code;  // For the command line, ie. 'V' for -V.
  bool argument;  // If it takes an argument on the command line.  This is set
                  // based on whether argname is empty.
  
  mem::string argname; // The name of the argument for printing the description.
  mem::string desc; // One line description of what the option does.

  option(mem::string name, char code, mem::string argname, mem::string desc)
    : name(name), code(code), argument(!argname.empty()),
      argname(argname), desc(desc) {}

  virtual ~option() {}

  // Builds this option's contribution to the optstring argument of get_opt().
  virtual mem::string optstring() {
    if (code) {
      mem::string base;
      base.push_back(code);
      return argument ? base+":" : base;
    }
    else return "";
  }

  // Sets the contribution to the longopt array.
  virtual void longopt(c_option &o) {
    o.name=name.c_str();
    o.has_arg=argument ? 1 : 0;
    o.flag=0;
    o.val=0;
  }

  // Add to the dictionaries of options.
  virtual void add() {
    optionsMap[name]=this;
    if (code)
      codeMap[code]=this;
  }

  // Set the option from the command-line argument.  Return true if the option
  // was correctly parsed.
  virtual bool getOption() = 0;

  void error(string msg) {
    cerr << argv0 << ": " << msg << " -- ";
    if (code)
      cerr << code << ", ";
    cerr << name << endl;
  }

  // The "-f -outformat format" part of the option.
  virtual mem::string describeStart() {
    ostringstream ss;
    if (code)
      ss << "-" << code << ", ";
    ss << "-" << name;
    if (argument)
      ss << " " << argname;
    return ss.str();
  }

  // Outputs description of the command for the -help option.
  virtual void describe() {
    // Don't show the option if it has no desciption.
    if (!desc.empty()) {
      const unsigned int WIDTH=20;
      mem::string start=describeStart();
      cerr << std::left << std::setw(WIDTH) << start;
      if (start.size() >= WIDTH) {
        cerr << endl;
        cerr << std::left << std::setw(WIDTH) << "";
      }
      cerr << desc << endl;
    }
  }
};

const mem::string noarg;

struct setting : public option {
  types::ty *t;

  setting(mem::string name, char code, mem::string argname, mem::string desc,
          types::ty *t)
      : option(name, code, argname, desc), t(t) {}

  virtual void reset() = 0;

  virtual trans::access *buildAccess() = 0;

  // Add to the dictionaries of options and to the settings module.
  virtual void add() {
    option::add();

    settingsModule.e.addVar(symbol::trans(name), buildVarEntry());
  }

  varEntry *buildVarEntry() {
    return new varEntry(t, buildAccess());
  }
};

struct itemSetting : public setting {
  item defaultValue;
  item value;

  itemSetting(mem::string name, char code,
              mem::string argname, mem::string desc,
              types::ty *t, item defaultValue)
      : setting(name, code, argname, desc, t), defaultValue(defaultValue) {
    reset();
  }

  void reset() {
    value=defaultValue;
  }

  trans::access *buildAccess() {
    return new itemRefAccess(&(value));
  }
};

struct boolSetting : public itemSetting {
  boolSetting(mem::string name, char code, mem::string desc,
              bool defaultValue=false)
    : itemSetting(name, code, noarg, desc,
              types::primBoolean(), (item)defaultValue) {}

  bool getOption() {
    value=(item)true;
    return true;
  }

  option *negation(mem::string name) {
    struct negOption : public option {
      boolSetting &base;

      negOption(boolSetting &base, mem::string name)
        : option(name, 0, noarg, ""), base(base) {}

      bool getOption() {
        base.value=(item)false;
        return true;
      }
    };
    return new negOption(*this, name);
  }

  void add() {
    setting::add();
    negation("no"+name)->add();
    if (code) {
      mem::string nocode="no"; nocode.push_back(code);
      negation(nocode)->add();
    }
  }

  // Set several related boolean options at once.  Used for view and trap which
  // have batch and interactive settings.
  struct multiOption : public option {
    typedef mem::list<boolSetting *> setlist;
    setlist set;
    multiOption(mem::string name, char code, mem::string desc)
      : option(name, code, noarg, desc) {}

    void add(boolSetting *s) {
      set.push_back(s);
    }

    void setValue(bool value) {
      for (setlist::iterator s=set.begin(); s!=set.end(); ++s)
        (*s)->value=(item)value;
    }

    bool getOption() {
      setValue(true);
      return true;
    }

    option *negation(mem::string name) {
      struct negOption : public option {
        multiOption &base;

        negOption(multiOption &base, mem::string name)
          : option(name, 0, noarg, ""), base(base) {}

        bool getOption() {
          base.setValue(false);
          return true;
        }
      };
      return new negOption(*this, name);
    }

    void add() {
      option::add();
      negation("no"+name)->add();
      if (code) {
        mem::string nocode="no"; nocode.push_back(code);
        negation(nocode)->add();
      }

      for (multiOption::setlist::iterator s=set.begin(); s!=set.end(); ++s)
        (*s)->add();
    }
  };
};

typedef boolSetting::multiOption multiOption;

struct argumentSetting : public itemSetting {
  argumentSetting(mem::string name, char code,
                  mem::string argname, mem::string desc,
                  types::ty *t, item defaultValue)
    : itemSetting(name, code, argname, desc, t, defaultValue) 
  {
    assert(!argname.empty());
  }
};

struct stringSetting : public argumentSetting {
  stringSetting(mem::string name, char code,
                mem::string argname, mem::string desc,
                mem::string defaultValue)
    : argumentSetting(name, code, argname, desc,
              types::primString(), (item)defaultValue) {}

  bool getOption() {
    value=(item)(mem::string)optarg;
    return true;
  }
};

struct realSetting : public argumentSetting {
  realSetting(mem::string name, char code,
              mem::string argname, mem::string desc,
              double defaultValue)
    : argumentSetting(name, code, argname, desc,
                  types::primReal(), (item)defaultValue) {}

  bool getOption() {
    try {
      value=(item)lexical::cast<double>(optarg);
    } catch (lexical::bad_cast&) {
      error("option requires a real number as an argument");
      return false;
    }
    return true;
  }
};

struct pairSetting : public argumentSetting {
  pairSetting(mem::string name, char code,
              mem::string argname, mem::string desc,
              camp::pair defaultValue=0.0)
    : argumentSetting(name, code, argname, desc,
                  types::primPair(), (item)defaultValue) {}

  bool getOption() {
    try {
      value=(item)lexical::cast<camp::pair>(optarg);
    } catch (lexical::bad_cast&) {
      error("option requires a pair as an argument");
      return false;
    }
    return true;
  }
};

// For setting the position of a figure on the page.
struct positionSetting : public argumentSetting {
  positionSetting(mem::string name, char code,
                  mem::string argname, mem::string desc,
                  int defaultValue=CENTER)
    : argumentSetting(name, code, argname, desc,
                  types::primInt(), (item)defaultValue) {}

  bool getOption() {
    mem::string str=optarg;
    if (str=="C")
      value=(int)CENTER;
    else if (str=="T")
      value=(int)TOP;
    else if (str=="B")
      value=(int)BOTTOM;
    else if (str=="Z") {
      value=(int)ZERO;
      getSetting("tex")=false;
    }
    else {
      error("invalid argument for option");
      return false;
    }
    return true;
  }
};

template <class T>
struct refSetting : public setting {
  T *ref;
  T defaultValue;

  refSetting(mem::string name, char code, mem::string argname, mem::string desc,
          types::ty *t, T *ref, T defaultValue)
      : setting(name, code, argname, desc, t),
        ref(ref), defaultValue(defaultValue) {
    reset();
  }

  virtual void reset() {
    *ref=defaultValue;
  }

  trans::access *buildAccess() {
    return new refAccess<T>(ref);
  }
};

#if INTV
struct incrementSetting : public refSetting<int> {
  incrementSetting(mem::string name, char code, mem::string desc, int *ref)
    : refSetting<int>(name, code, noarg, desc,
              types::primInt(), ref, 0) {}

  bool getOption() {
    // Increment the value.
    ++(*ref);
    return true;
  }
};
#else
struct incrementSetting : public itemSetting {
  incrementSetting(mem::string name, char code, mem::string desc)
    : itemSetting(name, code, noarg, desc,
              types::primInt(), (item)0) {}

  bool getOption() {
    // Increment the value in the item.
    int n=vm::get<int>(value);
    value=(item)(n+1);
    return true;
  }
};
#endif

void addOption(option *o) {
  o->add();
}

item &getSetting(string name) {
  itemSetting *s=dynamic_cast<itemSetting *>(optionsMap[name]);
  assert(s);
  return s->value;
}
  
void usage(const char *program)
{
  cerr << PROGRAM << " version " << VERSION
       << " [(C) 2004 Andy Hammerlindl, John C. Bowman, Tom Prince]" 
       << endl
       << "\t\t\t" << "http://sourceforge.net/projects/asymptote/"
       << endl
       << "Usage: " << program << " [options] [file ...]"
       << endl;
}

void reportSyntax() {
  cerr << endl;
  usage(argv0);
  cerr << endl << "Type '" << argv0
    << " -h' for a descriptions of options." << endl;
  exit(1);
}

void displayOptions()
{
  cerr << endl;
  cerr << "Options: " << endl;
  for (optionsMap_t::iterator opt=optionsMap.begin();
       opt!=optionsMap.end();
       ++opt)
    opt->second->describe();
}

struct helpOption : public option {
  helpOption(mem::string name, char code, mem::string desc)
    : option(name, code, noarg, desc) {}

  bool getOption() {
    usage(argv0);
    displayOptions();
    cerr << endl;
    exit(0);

    // Unreachable code.
    return true;
  }
};

// For security reasons, safe isn't a field of the settings module.
struct safeOption : public option {
  bool value;

  safeOption(mem::string name, char code, mem::string desc, bool value)
    : option(name, code, noarg, desc), value(value) {}

  bool getOption() {
    safe=value;
    return true;
  }
};


mem::string build_optstring() {
  mem::string s;
  for (codeMap_t::iterator p=codeMap.begin(); p !=codeMap.end(); ++p)
    s +=p->second->optstring();

  return s;
}

c_option *build_longopts() {
  int n=optionsMap.size();

#ifdef USEGC
  c_option *longopts=new (GC) c_option[n];
#else
  c_option *longopts=new c_option[n];
#endif

  int i=0;
  for (optionsMap_t::iterator p=optionsMap.begin();
       p !=optionsMap.end();
       ++p, ++i)
    p->second->longopt(longopts[i]);

  return longopts;
}

void getOptions(int argc, char *argv[])
{
  bool syntax=false;

  mem::string optstring=build_optstring();
  //cerr << "optstring: " << optstring << endl;
  c_option *longopts=build_longopts();
  int long_index = 0;

  errno=0;
  for(;;) {
    int c = getopt_long_only(argc,argv,
                             optstring.c_str(), longopts, &long_index);
    if (c == -1)
      break;

    if (c == 0) {
      const char *name=longopts[long_index].name;
      //cerr << "long option: " << name << endl;
      if (!optionsMap[name]->getOption())
        syntax=true;
    }
    else if (codeMap.find(c) != codeMap.end()) {
      //cerr << "char option: " << (char)c << endl;
      if (!codeMap[c]->getOption())
        syntax=true;
    }
    else {
      syntax=true;
    }

    errno=0;
  }
  
  if (syntax)
    reportSyntax();
}

// The verbosity setting, a global variable.
#if INTV
int verbose=0;
#else
item *verboseItem=0;
#endif

item *debugItem=0;

void initSettings() {
  multiOption *view=new multiOption("View", 'V', "View output file");
  view->add(new boolSetting("batchView", 0,
                     "View output files in batch mode", false));
  view->add(new boolSetting("interactiveView", 0,
                     "View output in interactive mode", true));
  view->add(new boolSetting("oneFileView", 0,
                     "View output of one file (for drag-and-drop)", msdos));
  addOption(view);
  //cerr << "-nV, -nView\t Don't view output file" << endl;

  addOption(new realSetting("deconstruct", 'x', "magnification",
                     "Deconstruct into transparent GIF objects", 0.0));
  addOption(new boolSetting("clearGUI", 'c', "Clear GUI operations"));
  addOption(new boolSetting("ignoreGUI", 'i', "Ignore GUI operations"));
  addOption(new stringSetting("outformat", 'f', "format",
                     "Convert each output file to specified format", "eps"));
  addOption(new stringSetting("outname", 'o', "name",
                     "(First) output file name", ""));
  addOption(new helpOption("help", 'h', "Show summary of options"));

  addOption(new pairSetting("offset", 'O', "pair", "PostScript offset"));
  addOption(new positionSetting("position", 'P', "[CBTZ]",
                     "Position of the figure on the page (Z implies -L)."));
  
  addOption(new boolSetting("debug", 'd', "Enable debugging messages"));
  debugItem=&getSetting("debug");

#if INTV
  addOption(new incrementSetting("verbose", 'v',
                     "Increase verbosity level", &verbose));
#else
  addOption(new incrementSetting("verbose", 'v',
                     "Increase verbosity level"));
  verboseItem=&getSetting("verbose");
#endif

  addOption(new boolSetting("keep", 'k', "Keep intermediate files"));
  addOption(new boolSetting("tex", 0,
                     "Enable LaTeX label postprocessing (default)", true));
  //cerr << "-L\t\t Disable LaTeX label postprocessing" << endl;
  addOption(new boolSetting("texmode", 't',
                     "Produce LaTeX file for \\usepackage[inline]{asymptote}"));
  addOption(new boolSetting("parseonly", 'p', "Parse test"));
  addOption(new boolSetting("translate", 's', "Translate test"));
  addOption(new boolSetting("listvariables", 'l',
                     "List available global functions and variables"));
  
  multiOption *mask=new multiOption("mask", 'm',
                        "Mask fpu exceptions");
  mask->add(new boolSetting("batchMask", 0,
                     "Mask fpu exceptions in batch mode", false));
  mask->add(new boolSetting("interactiveMask", 0,
                     "Mask fpu exceptions in interactive mode", true));
  addOption(mask);

  addOption(new boolSetting("bw", 0,
                     "Convert all colors to black and white"));
  addOption(new boolSetting("gray", 0, "Convert all colors to grayscale"));
  addOption(new boolSetting("rgb", 0, "Convert cmyk colors to rgb"));
  addOption(new boolSetting("cmyk", 0, "Convert rgb colors to cmyk"));

  addOption(new safeOption("safe", 0,
                    "Disable system call (default, negation ignored)", true));
  addOption(new safeOption("unsafe", 0,
                    "Enable system call (negation ignored)", false));

  addOption(new boolSetting("localhistory", 0, 
                     "Use a local interactive history file"));
  addOption(new boolSetting("autoplain", 0,
                     "Enable automatic importing of plain (default)", true));
  //cerr << "-noplain\t Disable automatic importing of plain" << endl;
}



#ifdef MSDOS
const char pathSeparator=';';
//int view=-1; // Support drag and drop in MSWindows
const string defaultPSViewer=
  "'c:\\Program Files\\Ghostgum\\gsview\\gsview32.exe'";
const string defaultPDFViewer=
  "'c:\\Program Files\\Adobe\\Acrobat 7.0\\Reader\\AcroRd32.exe'";
const string defaultGhostscript=
  "'c:\\Program Files\\gs\\gs8.51\\bin\\gswin32.exe'";
const string defaultPython="'c:\\Python24\\python.exe'";
const string defaultDisplay="imdisplay";
#undef ASYMPTOTE_SYSDIR
#define ASYMPTOTE_SYSDIR "c:\\Program Files\\Asymptote"
const string docdir=".";
#else  
const char pathSeparator=':';
//int view=0;
const string defaultPSViewer="gv";
const string defaultPDFViewer="acroread";
const string defaultGhostscript="gs";
const string defaultDisplay="display";
const string defaultPython="";
const string docdir=ASYMPTOTE_DOCDIR;
#endif  
  
string PSViewer;
string PDFViewer;
string Ghostscript;
string LaTeX;
string Dvips;
string Convert;
string Display;
string Animate;
string Python;
string Xasy;
  
int safe=1;
  
int ShipoutNumber=0;
string paperType;
double pageWidth;
double pageHeight;

int scrollLines=0;
  
const string suffix="asy";
const string guisuffix="gui";
  

bool TeXinitialized=false;
string initdir;

camp::pen defaultpen=camp::pen::startupdefaultpen();
  

// Local versions of the argument list.
int argCount = 0;
char **argList = 0;
  
// Access the arguments once options have been parsed.
int numArgs() { return argCount; }
char *getArg(int n) { return argList[n]; }

void setInteractive() {
  if(numArgs() == 0 && !getSetting<bool>("listvariables")) {
    interact::interactive=true;

    // NOTE: Move this greeting to the start of the interactive prompt.  It has
    // nothing to do with settings.
    cout << "Welcome to " << PROGRAM << " version " << VERSION
	 << " (to view the manual, type help)" << endl;
  }
}

bool view() {
  if (interact::interactive)
    return getSetting<bool>("interactiveView");
  else
    return getSetting<bool>("batchView") ||
           (numArgs()==1 && getSetting<bool>("oneFileView"));
}

bool trap() {
  if (interact::interactive)
    return !getSetting<bool>("interactiveMask");
  else
    return !getSetting<bool>("batchMask");
}

void setOptionsFromFile() {
  std::ifstream finit;
  initdir=Getenv("HOME",false)+"/.asy";
  mkdir(initdir.c_str(),0xFFFF);
  initdir += "/";
  finit.open((initdir+"options").c_str());
	
  if(finit) {
    string s;
    ostringstream buf;
    vector<string> Args;
    while(finit >> s)
      Args.push_back(s);
    finit.close();
    
    int Argc=(int) Args.size()+1;
    char** Argv=new char*[Argc];
    Argv[0]=argv0;
    int i=1;
    
    for(vector<string>::iterator p=Args.begin(); p != Args.end(); ++p)
      Argv[i++]=strcpy(new char[p->size()+1],p->c_str());
    
    getOptions(Argc,Argv);
    delete[] Argv;
    optind=0;
  }
}

void setPath() {
  searchPath.push_back(".");
  string asydir=Getenv("ASYMPTOTE_DIR",false);
  if(asydir != "") {
    size_t p,i=0;
    while((p=asydir.find(pathSeparator,i)) < string::npos) {
      if(p > i) searchPath.push_back(asydir.substr(i,p-i));
      i=p+1;
    }
    if(i < asydir.length()) searchPath.push_back(asydir.substr(i));
  }
#ifdef ASYMPTOTE_SYSDIR
  searchPath.push_back(ASYMPTOTE_SYSDIR);
#endif
}

void setApplicationNames() {
  string psviewer=Getenv("ASYMPTOTE_PSVIEWER");
  string pdfviewer=Getenv("ASYMPTOTE_PDFVIEWER");
  string ghostscript=Getenv("ASYMPTOTE_GS");
  string latex=Getenv("ASYMPTOTE_LATEX");
  string dvips=Getenv("ASYMPTOTE_DVIPS");
  string convert=Getenv("ASYMPTOTE_CONVERT");
  string display=Getenv("ASYMPTOTE_DISPLAY");
  string animate=Getenv("ASYMPTOTE_ANIMATE");
  string python=Getenv("ASYMPTOTE_PYTHON");
  string xasy=Getenv("ASYMPTOTE_XASY");

  PSViewer=psviewer != "" ? psviewer : defaultPSViewer;
  PDFViewer=pdfviewer != "" ? pdfviewer : defaultPDFViewer;
  Ghostscript=ghostscript != "" ? ghostscript : defaultGhostscript;
  LaTeX=latex != "" ? latex : "latex";
  Dvips=dvips != "" ? dvips : "dvips";
  Convert=convert != "" ? convert : "convert";
  Display=display != "" ? display : defaultDisplay;
  Animate=animate != "" ? animate : "animate";
  Python=python != "" ? python : defaultPython;
  Xasy=xasy != "" ? xasy : "xasy";
}

void setPaperType() {
  char *papertype=getenv("ASYMPTOTE_PAPERTYPE");
  paperType=papertype ? papertype : "letter";

  if(paperType == "letter") {
    pageWidth=72.0*8.5;
    pageHeight=72.0*11.0;
  } else {
    pageWidth=72.0*21.0/2.54;
    pageHeight=72.0*29.7/2.54;
    if(paperType != "a4") {
      cerr << "Unknown paper size \'" << paperType << "\'; assuming a4." 
	   << endl;
      paperType="a4";
    }
  }
}

void setOptions(int argc, char *argv[])
{
  argv0=argv[0];

  // Build settings module.
  initSettings();
  
  // Set options given in $HOME/.asy/options
  setOptionsFromFile();
  
  getOptions(argc,argv);
  
  // Set variables for the normal arguments.
  argCount = argc - optind;
  argList = argv + optind;

#if 0
  if(origin == ZERO)
    getSetting("texprocess")=false;
#endif
  
  setPath();
  setApplicationNames();

#ifdef MSDOS
  if(!Getenv(CYGWIN)) newline="\r\n";
#endif
  
  setPaperType();

  setInteractive();
}

}
