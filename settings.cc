/*****
 * settings.cc
 * Andy Hammerlindl 2004/05/10
 *
 * Declares a list of global variables that act as settings in the system.
 *****/

#include <iostream>
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

struct option : public gc {
  mem::string name;
  char code;  // For the command line, ie. 'V' for -V.
  bool argument;  // If it takes an argument on the command line.

  mem::string desc; // One line description of what the option does.

  option(mem::string name, char code, bool argument, mem::string desc)
    : name(name), code(code), argument(argument), desc(desc) {}

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

  // Set the option from the command-line argument.
  virtual void getOption() = 0;
};

struct setting : public option {
  types::ty *t;

  setting(mem::string name, char code, bool argument, mem::string desc,
          types::ty *t)
      : option(name, code, argument, desc), t(t) {}

  virtual void reset() = 0;

  virtual trans::access *buildAccess() = 0;

  varEntry *buildVarEntry() {
    return new varEntry(t, buildAccess());
  }
};

struct itemSetting : public setting {
  item defaultValue;
  item value;

  itemSetting(mem::string name, char code, bool argument, mem::string desc,
              types::ty *t, item defaultValue)
      : setting(name, code, argument, desc, t), defaultValue(defaultValue) {
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
    : itemSetting(name, code, false, desc,
              types::primBoolean(), (item)defaultValue) {}

  static bool negate;
  struct negateOption : public option {
    negateOption()
      : option("no", 'n', false, "Negate next option") {}

    void getOption() {
      negate=true;
    }
  };

  void getOption() {
    if (negate) {
      value=(item)false;
      negate=false;
    }
    else
      value=(item)true;
  }

  // Set several related boolean options at once.  Used for view and trap which
  // have batch and interactive settings.
  struct multiOption : public option {
    typedef mem::list<boolSetting *> setlist;
    setlist set;
    multiOption(mem::string name, char code, mem::string desc)
      : option(name, code, false, desc) {}

    void add(boolSetting *s) {
      set.push_back(s);
    }

    void setValue(bool value) {
      for (setlist::iterator s=set.begin(); s!=set.end(); ++s)
        (*s)->value=(item)value;
    }

    void getOption() {
      if (negate) {
        setValue(false);
        negate=false;
      }
      else
        setValue(true);
    }
  };
};

bool boolSetting::negate=false;
typedef boolSetting::multiOption multiOption;

struct stringSetting : public itemSetting {
  stringSetting(mem::string name, char code,
                mem::string /*argname*/, mem::string desc,
                mem::string defaultValue)
    : itemSetting(name, code, true, desc,
              types::primString(), (item)defaultValue) {}

  void getOption() {
    value=(item)(mem::string)optarg;
  }
};

struct realSetting : public itemSetting {
  realSetting(mem::string name, char code,
              mem::string /*argname*/, mem::string desc,
              double defaultValue)
    : itemSetting(name, code, true, desc,
                  types::primReal(), (item)defaultValue) {}

  void getOption() {
    try {
      value=(item)lexical::cast<double>(optarg);
    } catch (lexical::bad_cast&) {
      cerr << "Setting '" << name << "' is a real number." << endl;
    }
  }
};

struct pairSetting : public itemSetting {
  pairSetting(mem::string name, char code,
              mem::string /*argname*/, mem::string desc,
              camp::pair defaultValue=0.0)
    : itemSetting(name, code, true, desc,
                  types::primPair(), (item)defaultValue) {}

  void getOption() {
    try {
      value=(item)lexical::cast<camp::pair>(optarg);
    } catch (lexical::bad_cast&) {
      cerr << "Setting '" << name << "' is a pair." << endl;
    }
  }
};

// For setting the position of a figure on the page.
struct positionSetting : public itemSetting {
  positionSetting(mem::string name, char code,
                  mem::string /*argname*/, mem::string desc,
                  int defaultValue=CENTER)
    : itemSetting(name, code, true, desc,
                  types::primInt(), (item)defaultValue) {}

  void getOption() {
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
    else 
      cerr << "Invalid argument for setting " << name << "." << endl;
  }
};

template <class T>
struct refSetting : public setting {
  T *ref;
  T defaultValue;

  refSetting(mem::string name, char code, bool argument, mem::string desc,
          types::ty *t, T *ref, T defaultValue)
      : setting(name, code, argument, desc, t),
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
    : refSetting<int>(name, code, false, desc,
              types::primInt(), ref, 0) {}

  void getOption() {
    // Increment the value.
    ++(*ref);
  }
};
#else
struct incrementSetting : public itemSetting {
  incrementSetting(mem::string name, char code, mem::string desc)
    : itemSetting(name, code, false, desc,
              types::primInt(), (item)0) {}

  void getOption() {
    // Increment the value in the item.
    int n=vm::get<int>(value);
    value=(item)(n+1);
  }
};
#endif

typedef mem::map<const mem::string, option *> optionsMap_t;
optionsMap_t optionsMap;
typedef mem::map<const char, option *> codeMap_t;
codeMap_t codeMap;
  
void addOption(option *o) {
  optionsMap[o->name]=o;
  if (o->code)
    codeMap[o->code]=o;
}

void addSetting(setting *s) {
  addOption(s);

  settingsModule.e.addVar(symbol::trans(s->name), s->buildVarEntry());
}

void addMultiOption(multiOption *m) {
  addOption(m);

  for (multiOption::setlist::iterator s=m->set.begin(); s!=m->set.end(); ++s)
    addSetting(*s);
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
  cerr << "-V, -View\t View output file" << endl;
  cerr << "-nV, -nView\t Don't view output file" << endl;
  cerr << "-n, -no\t\t Negate next option" << endl;
  cerr << "-x magnification Deconstruct into transparent GIF objects" << endl;
  cerr << "-c \t\t Clear GUI operations" << endl;
  cerr << "-i \t\t Ignore GUI operations" << endl;
  cerr << "-f format\t Convert each output file to specified format" << endl;
  cerr << "-o name\t\t (First) output file name" << endl;
  cerr << "-h, -help\t Show summary of options" << endl;
  cerr << "-O pair\t\t PostScript offset" << endl; 
  cerr << "-C\t\t Center on page (default)" << endl;
  cerr << "-B\t\t Align to bottom-left corner of page" << endl;
  cerr << "-T\t\t Align to top-left corner of page" << endl;
  cerr << "-Z\t\t Position origin at (0,0) (implies -L)" << endl;
  cerr << "-d\t\t Enable debugging messages" << endl;
  cerr << "-v, -verbose\t Increase verbosity level" << endl;
  cerr << "-k\t\t Keep intermediate files" << endl;
  cerr << "-L\t\t Disable LaTeX label postprocessing" << endl;
  cerr << "-t\t\t Produce LaTeX file for \\usepackage[inline]{asymptote}"
       << endl;
  cerr << "-p\t\t Parse test" << endl;
  cerr << "-s\t\t Translate test" << endl;
  cerr << "-l\t\t List available global functions and variables" << endl;
  cerr << "-m, -mask\t Mask fpu exceptions (default for interactive mode)"
       << endl;
  cerr << "-nm, -nmask\t Don't mask fpu exceptions (default for batch mode)" 
       << endl;
  cerr << "-bw\t\t Convert all colors to black and white" << endl;
  cerr << "-gray\t\t Convert all colors to grayscale" << endl;
  cerr << "-rgb\t\t Convert cmyk colors to rgb" << endl;
  cerr << "-cmyk\t\t Convert rgb colors to cmyk" << endl;
  cerr << "-safe\t\t Disable system call (default, negation ignored)" << endl;
  cerr << "-unsafe\t\t Enable system call (negation ignored)" << endl;
  cerr << "-localhistory\t Use a local interactive history file"
       << endl;
  cerr << "-noplain\t Disable automatic importing of plain" << endl;
}

struct helpOption : public option {
  helpOption(mem::string name, char code, mem::string desc)
    : option(name, code, false, desc) {}

  void getOption() {
    usage(argv0);
    displayOptions();
    cerr << endl;
    exit(0);
  }
};

// For security reasons, safe isn't a field of the settings module.
struct safeOption : public option {
  bool value;

  safeOption(mem::string name, char code, mem::string desc, bool value)
    : option(name, code, false, desc), value(value) {}

  void getOption() {
    safe=value;
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
      optionsMap[name]->getOption();
    }
    else if (codeMap.find(c) != codeMap.end()) {
      //cerr << "char option: " << (char)c << endl;
      codeMap[c]->getOption();
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
  addMultiOption(view);
  //cerr << "-nV, -nView\t Don't view output file" << endl;

  addOption(new boolSetting::negateOption);

  addSetting(new realSetting("deconstruct", 'x', "magnification",
                     "Deconstruct into transparent GIF objects", 0.0));
  addSetting(new boolSetting("clearGUI", 'c', "Clear GUI operations"));
  addSetting(new boolSetting("ignoreGUI", 'i', "Ignore GUI operations"));
  addSetting(new stringSetting("outformat", 'f', "format",
                     "Convert each output file to specified format", "eps"));
  addSetting(new stringSetting("outname", 'o', "name",
                     "(First) output file name", ""));
  addOption(new helpOption("help", 'h', "Show summary of options"));

  addSetting(new pairSetting("offset", 'O', "pair", "PostScript offset"));
  addSetting(new positionSetting("position", 'P', "[CBTZ]",
                     "Position of the figure on the page (Z implies -L)."));
  
  addSetting(new boolSetting("debug", 'd', "Enable debugging messages"));
  debugItem=&getSetting("debug");

#if INTV
  addSetting(new incrementSetting("verbose", 'v',
                     "Increase verbosity level", &verbose));
#else
  addSetting(new incrementSetting("verbose", 'v',
                     "Increase verbosity level"));
  verboseItem=&getSetting("verbose");
#endif

  addSetting(new boolSetting("keep", 'k', "Keep intermediate files"));
  addSetting(new boolSetting("tex", 0,
                     "Enable LaTeX label postprocessing (default)", true));
  //cerr << "-L\t\t Disable LaTeX label postprocessing" << endl;
  addSetting(new boolSetting("texmode", 't',
                     "Produce LaTeX file for \\usepackage[inline]{asymptote}"));
  addSetting(new boolSetting("parseonly", 'p', "Parse test"));
  addSetting(new boolSetting("translate", 's', "Translate test"));
  addSetting(new boolSetting("listvariables", 'l',
                     "List available global functions and variables"));
  
  multiOption *mask=new multiOption("mask", 'm',
                        "Mask fpu exceptions");
  mask->add(new boolSetting("batchMask", 0,
                     "Mask fpu exceptions in batch mode", false));
  mask->add(new boolSetting("interactiveMask", 0,
                     "Mask fpu exceptions in interactive mode", true));
  addMultiOption(mask);

  addSetting(new boolSetting("bw", 0,
                     "Convert all colors to black and white"));
  addSetting(new boolSetting("gray", 0, "Convert all colors to grayscale"));
  addSetting(new boolSetting("rgb", 0, "Convert cmyk colors to rgb"));
  addSetting(new boolSetting("cmyk", 0, "Convert rgb colors to cmyk"));

  addOption(new safeOption("safe", 0,
                    "Disable system call (default, negation ignored)", true));
  addOption(new safeOption("unsafe", 0,
                    "Enable system call (negation ignored)", false));

  addSetting(new boolSetting("localhistory", 0, 
                     "Use a local interactive history file"));
  addSetting(new boolSetting("autoplain", 0,
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
