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

using std::vector;

namespace settings {
  
const char PROGRAM[]=PACKAGE_NAME;
const char VERSION[]=PACKAGE_VERSION;
const char BUGREPORT[]=PACKAGE_BUGREPORT;

#ifdef MSDOS
const char pathSeparator=';';
int view=-1; // Support drag and drop in MSWindows
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
int view=0;
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
  
string outformat="eps";
int keep=0;
int texprocess=1;
int texmode=0;
int debug=0;
int verbose=0;
int safe=1;
int autoplain=1;
int localhistory=0;
int parseonly=0;
int translate=0;
int listvariables=0;
int bwonly=0;
int grayonly=0;
int rgbonly=0;
int cmykonly=0;
int trap=-1;
double deconstruct=0;
int clearGUI=0;
int ignoreGUI=0;
camp::pair postscriptOffset=0.0;
int origin=CENTER;
  
int ShipoutNumber=0;
string paperType;
double pageWidth;
double pageHeight;

int scrollLines=0;
  
const string suffix="asy";
const string guisuffix="gui";
  
string outname;

bool TeXinitialized=false;
string initdir;

camp::pen defaultpen=camp::pen::startupdefaultpen();
  
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
  cerr << "-nV, -nView\t Don't view output file" << endl;
  cerr << "-n, -no\t\t Negate next option" << endl;
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

// Local versions of the argument list.
int argCount = 0;
char **argList = 0;
  
  // Access the arguments once options have been parsed.
int numArgs() { return argCount; }
char *getArg(int n) { return argList[n]; }

int no=0;
  
int set() {
  if(no) {no=0; return 0;}
  return 1;
}
  
void getOptions(int argc, char *argv[])
{
  int syntax=0;
  int option_index = 0;

  enum Options {BW=257,GRAY,RGB,CMYK,NOPLAIN,LOCALHISTORY};
  
  static struct option long_options[] =
  {
    {"verbose", 0, 0, 'v'},
    {"help", 0, 0, 'h'},
    {"View", 0, 0, 'V'},
    {"mask", 0, 0, 'm'},
    {"no", 0, 0, 'n'},
    {"bw", 0, 0, BW},
    {"gray", 0, 0, GRAY},
    {"rgb", 0, 0, RGB},
    {"cmyk", 0, 0, CMYK},
    {"noplain", 0, 0, NOPLAIN},
    {"localhistory", 0, 0, LOCALHISTORY},
    {"safe", 0, &safe, 1},
    {"unsafe", 0, &safe, 0},
    {0, 0, 0, 0}
  };

 errno=0;
  for(;;) {
    int c = getopt_long_only(argc,argv,
			     "cdf:hiklLmo:pPstvVnx:O:CBTZ",
			     long_options,&option_index);
    if (c == -1) break;

    switch (c) {
    case 0:
      break;
    case 'c':
      clearGUI=set();
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
      ignoreGUI=set();
      break;
    case 'k':
      keep=set();
      break;
    case 'L':
      texprocess=!set();
      break;
    case 'm': 
      trap=!set();
      break;
    case 'p':
      parseonly=set();
      break;
    case 'o':
      outname=string(optarg);
      break;
    case 's':
      translate=set();
      break;
    case 't':
      texmode=set();
      break;
    case 'l':
      listvariables=set();
      break;
    case 'd':
      debug=set();
      break;
    case 'v':
      verbose += set() ? 1 : -1;
      break;
    case 'V':
      view=set();
      break;
    case 'n':
      no=1;
      break;
    case 'x':
      try {
        deconstruct=lexical::cast<double>(optarg);
      } catch (lexical::bad_cast&) {
        syntax=1;
      }
      if(deconstruct < 0) syntax=1;
      break;
    case 'O':
      try {
        postscriptOffset=lexical::cast<camp::pair>(optarg);
      } catch (lexical::bad_cast&) {
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
    case BW:
      bwonly=set();
      break;
    case GRAY:
      grayonly=set();
      break;
    case RGB:
      rgbonly=set();
      break;
    case CMYK:
      cmykonly=set();
      break;
    case NOPLAIN:
      autoplain=!set();
      break;
    case LOCALHISTORY:
      localhistory=set();
      break;
    default:
      syntax=1;
    }
  errno=0;
  }
  
  if (syntax) {
    cerr << endl;
    usage(argv[0]);
    cerr << endl << "Type '" << argv[0]
	 << " -h' for a descriptions of options." << endl;
    exit(1);
  }
}

void setOptions(int argc, char *argv[])
{
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
    Argv[0]=argv[0];
    int i=1;
    
    for(vector<string>::iterator p=Args.begin(); p != Args.end(); ++p)
      Argv[i++]=strcpy(new char[p->size()+1],p->c_str());
    
    getOptions(Argc,Argv);
    delete[] Argv;
    optind=0;
  }
  
  getOptions(argc,argv);
  
  // Set variables for the normal arguments.
  argCount = argc - optind;
  argList = argv + optind;

  if(numArgs() == 0 && !listvariables) {
    view=1;
    interact::interactive=true;
    if(trap == -1) trap=0;
    cout << "Welcome to " << PROGRAM << " version " << VERSION
	 << " (to view the manual, type help)" << endl;
  } else if(trap == -1) trap=1;

  if(view == -1) view=(numArgs() == 1) ? 1 : 0; 
  
  if(origin == ZERO) texprocess=0;
  
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

}
