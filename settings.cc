/*****
 * settings.cc
 * Andy Hammerlindl 2004/05/10
 *
 * Declares a list of global variables that act as settings in the system.
 *****/

#include <iostream>
#include <cstdlib>
#include <cerrno>
#include <boost/lexical_cast.hpp>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#if HAVE_GNU_GETOPT_H
#include <getopt.h>
#else
// Work around conflicts with non-GNU versions of getopt.h:
struct option {
  const char *name;
  int has_arg, *flag, val;
};
extern "C" int getopt_long_only(int argc, char * const argv[],
				const char *optstring,
				const struct option *longopts, int *longindex);
#endif

#include "util.h"
#include "settings.h"

using boost::lexical_cast;
using std::string;

namespace settings {
const char PROGRAM[]=PACKAGE_NAME;
const char VERSION[]=PACKAGE_VERSION;
const char BUGREPORT[]=PACKAGE_BUGREPORT;

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
camp::pair postscriptOffset=camp::pair(18,-18);
int bottomOrigin=0;
  
double defaultlinewidth=0.0;  
double defaultfontsize=0.0;
bool suppressOutput=false;
bool upToDate=false;
int overwrite=0;
bool defaultOrigin=true;

int ShipoutNumber=0;
char* AsyDir;
string PSViewer;
string PDFViewer;

const std::string suffix="asy";
const std::string guisuffix="gui";
  
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
  cerr << "-i \t\t ignore GUI operations" << endl;
  cerr << "-x magnification deconstruct into transparent gif objects" 
       << endl;
  cerr << "-f format\t convert each output file to specified format" << endl;
  cerr << "-V, -View\t view output file" << endl;
  cerr << "-h, -help\t help" << endl;
  cerr << "-o name\t\t (first) output file name" << endl;
  cerr << "-L\t\t disable LaTeX label postprocessing" << endl;
  cerr << "-O pair\t\t PostScript offset: defaults to (18,-18)"
       << endl; 
  cerr << "-b\t\t align to bottom-left (instead of top-left) corner of page"
       << endl;
  cerr << "-v, -verbose\t increase verbosity level" << endl;
  cerr << "-k\t\t keep intermediate files" << endl;
  cerr << "-p\t\t parse test" << endl;
  cerr << "-s\t\t translate test" << endl;
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
			     "bcf:hikLmo:pPsvVx:O:",
			     long_options,&option_index);
    if (c == -1) break;

    switch (c) {
    case 0:
      break;
    case 'b':
      bottomOrigin=1;
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
        postscriptOffset=lexical_cast<camp::pair>(optarg);
	defaultOrigin=false;
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
  
  AsyDir=getenv("ASYMPTOTE_DIR");
  char *psviewer=getenv("ASYMPTOTE_PSVIEWER");
  char *pdfviewer=getenv("ASYMPTOTE_PDFVIEWER");
  PSViewer=psviewer ? psviewer : "gv";
  PDFViewer=pdfviewer ? pdfviewer : "gv";
  
  if(defaultOrigin && bottomOrigin) postscriptOffset=conj(postscriptOffset);
}

// Reset to startup defaults
void reset() 
{
  defaultfontsize=12.0;
  defaultlinewidth=0.5;  
  overwrite=0;
}

}
