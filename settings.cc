/*****
 * settings.cc
 * Andy Hammerlindl 2004/05/10
 *
 * Declares a list of global variables that act as settings in the system.
 *****/

#include <iostream>
#include <cstdlib>
#include <getopt.h>
#include <cerrno>
#include <boost/lexical_cast.hpp>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "util.h"
#include "settings.h"

using boost::lexical_cast;
using std::string;

namespace settings {
const char PROGRAM[]=PACKAGE_NAME;
const char VERSION[]=PACKAGE_VERSION;

string outformat="eps";
int keep=0;
int texprocess=1;
int verbose=0;
int view=0;
int safe=1;
int autoplain=1;
int parseonly=0;
int translate=0;
int trap=1;
double deconstruct=0;
int clearGUI=0;
int ignoreGUI=0;
camp::pair printerOffset=camp::pair(0,0);
  
double defaultlinewidth=0.0;  
double defaultfontsize=0.0;
bool suppressOutput=false;
bool upToDate=false;
int overwrite=0;

int ShipoutNumber=0;
char localdir[]=".";

std::string suffix="asy";
  
const char *getAsyDir() {
  char *dir = getenv("ASYMPTOTE_DIR");
  return dir ? dir : localdir;
}

string outname;
std::list<string> *outnameStack;

bool TeXinitialized=false;

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

void options()
{
  cerr << endl;
  cerr << "Options: " << endl;
  cerr << "-c \t\t clear GUI operations" << endl;
  cerr << "-x magnification deconstruct into transparent gif objects" 
       << endl;
  cerr << "-f format\t convert each output file to specified format" << endl;
  cerr << "-h, -help\t help" << endl;
  cerr << "-i \t\t ignore GUI operations" << endl;
  cerr << "-k\t\t keep intermediate files" << endl;
  cerr << "-L\t\t disable LaTeX label postprocessing" << endl;
  cerr << "-p\t\t parse test" << endl;
  cerr << "-O value\t real or pair printer offset (postscript pt)" << endl;
  cerr << "-o name\t\t (first) output file name" << endl;
  cerr << "-s\t\t translate test" << endl;
  cerr << "-v, -verbose\t increase verbosity level" << endl;
  cerr << "-V, -View\t view output file" << endl;
  cerr << "-m\t\t mask fpu exceptions (on supported architectures)" << endl;
  cerr << "-nomask\t\t don't mask fpu exceptions (default)" << endl;
  cerr << "-safe\t\t disable system call (default)" << endl;
  cerr << "-unsafe\t\t enable system call" << endl;
  cerr << "-noplain\t disable automatic importing of plain" << endl;
}

// Local versions of the argument list.
int argCount = 0;
char **argList = 0;
  
  // Access the arguments once options have been parsed.
int numArgs() { return argCount; }
char *getArg(int n) { return argList[n]; }

void setOptions(int argc, char *argv[])
{
  int syntax=0;
  int option_index = 0;

  static struct option long_options[] =
  {
    {"verbose", 0, 0, 'v'},
    {"help", 0, 0, 'h'},
    {"safe", 0, &safe, 1},
    {"unsafe", 0, &safe, 0},
    {"View", 0, &view, 1},
    {"mask", 0, &trap, 0},
    {"nomask", 0, &trap, 1},
    {"noplain", 0, &autoplain, 0},
    {0, 0, 0, 0}
  };

  errno=0;
  for(;;) {
    int c = getopt_long_only(argc,argv,
			     "cf:hikLmo:pPsvVx:O:",
			     long_options,&option_index);
    if (c == -1) break;

    switch (c) {
    case 0:
      break;
    case 'c':
      clearGUI=1;
      break;
    case 'f':
      outformat=string(optarg);
      break;
    case 'h':
      usage(argv[0]);
      options();
      cerr << endl;
      exit(0);
    case 'i':
      ignoreGUI=1;
      break;
    case 'k':
      keep=1;
      break;
    case 'L':
      texprocess=0;
      break;
    case 'm': 
      trap=0;
      break;
    case 'p':
      parseonly=1;
      break;
    case 'o':
      outname=string(optarg);
      break;
    case 's':
      translate=1;
      break;
    case 'v':
      verbose++;
      break;
    case 'V':
      view=1;
      break;
    case 'x':
      try {
        deconstruct=lexical_cast<double>(optarg);
      } catch (boost::bad_lexical_cast&) {
        syntax=1;
      }
      if(deconstruct <= 0) syntax=1;
      break;
    case 'O':
      try {
        printerOffset=lexical_cast<camp::pair>(optarg);
	if(printerOffset.isreal()) 
	  printerOffset=camp::pair(printerOffset.getx(),printerOffset.getx());
      } catch (boost::bad_lexical_cast&) {
        syntax=1;
      }
      break;
    default:
      syntax=1;
    }
  }

  errno=0;

  // Set variables for the normal arguments.
  argCount = argc - optind;
  argList = argv + optind;

  if (syntax) {
    cerr << endl;
    usage(argv[0]);
    cerr << endl << "Type '" << argv[0]
	 << " -h' for a descriptions of options." << endl;
    exit(1);
  }
}

// Reset to startup defaults
void reset() 
{
  defaultfontsize=12.0;
  defaultlinewidth=0.5;  
  overwrite=0;
}

}
