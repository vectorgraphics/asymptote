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
int rgbonly=0;
int cmykonly=0;
int trap=1;
double deconstruct=0;
int clearGUI=0;
int ignoreGUI=0;
camp::pair postscriptOffset=0.0;
int origin=CENTER;
  
double defaultlinewidth=0.0;  
double defaultfontsize=0.0;
bool suppressOutput=false;
bool upToDate=false;
int overwrite=0;

int ShipoutNumber=0;
char* AsyDir;
string PSViewer;
string PDFViewer;
string paperType;
double pageWidth;
double pageHeight;

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
  cerr << "-V, -View\t View output file" << endl;
  cerr << "-x magnification Deconstruct into transparent GIF objects" 
       << endl;
  cerr << "-c \t\t Clear GUI operations" << endl;
  cerr << "-i \t\t Ignore GUI operations" << endl;
  cerr << "-f format\t Convert each output file to specified format" << endl;
  cerr << "-o name\t\t (First) output file name" << endl;
  cerr << "-h, -help\t Show summary of options" << endl;
  cerr << "-O pair\t\t PostScript offset" << endl; 
  cerr << "-C\t\t Center on page (default)" << endl;
  cerr << "-B\t\t Align to bottom-left corner of page" << endl;
  cerr << "-T\t\t Align to top-left corner of page" << endl;
  cerr << "-Z\t\t Position origin at (0,0)" << endl;
  cerr << "-v, -verbose\t Increase verbosity level" << endl;
  cerr << "-k\t\t Keep intermediate files" << endl;
  cerr << "-L\t\t Disable LaTeX label postprocessing" << endl;
  cerr << "-p\t\t Parse test" << endl;
  cerr << "-s\t\t Translate test" << endl;
  cerr << "-m\t\t Mask fpu exceptions (on supported architectures)" << endl;
  cerr << "-nomask\t\t Don't mask fpu exceptions (default)" << endl;
  cerr << "-cmyk\t\t Convert rgb colors to cmyk" << endl;
  cerr << "-rgb\t\t Convert cmyk colors to rgb" << endl;
  cerr << "-safe\t\t Disable system call (default)" << endl;
  cerr << "-unsafe\t\t Enable system call" << endl;
  cerr << "-noplain\t Disable automatic importing of plain" << endl;
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
    {"rgb", 0, &rgbonly, 1},
    {"cmyk", 0, &cmykonly, 1},
    {"noplain", 0, &autoplain, 0},
    {0, 0, 0, 0}
  };

  errno=0;
  for(;;) {
    int c = getopt_long_only(argc,argv,
			     "cf:hikLmo:pPsvVx:O:CBTZ",
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
        postscriptOffset=lexical_cast<camp::pair>(optarg);
      } catch (boost::bad_lexical_cast&) {
        syntax=1;
      }
      break;
    case 'C':
      origin=CENTER;
      break;
    case 'B':
      origin=BOTTOM;
      break;
    case 'T':
      origin=TOP;
      break;
    case 'Z':
      origin=ZERO;
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

// Reset to startup defaults
void reset() 
{
  defaultfontsize=12.0;
  defaultlinewidth=0.5;  
  overwrite=0;
}

}
