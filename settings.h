/*****
 * settings.h
 * Andy Hammerlindl 2004/05/10
 *
 * Declares a list of global variables that act as settings in the system.
 *****/

#ifndef SETTINGS_H
#define SETTINGS_H

#include <string>
#include <list>
#include <fstream>

#include "pair.h"
#include "item.h"

using std::string;

// For testing various form of implementing verbose.
#define INTV 1
#if INTV
  #define VERBOSE (settings::verbose)
#else
  #define VERBOSE (settings::verbose())
#endif

namespace types {
  class record;
}

namespace settings {
extern const char PROGRAM[];
extern const char VERSION[];
extern const char BUGREPORT[];

extern string psviewer;    // Environment variable ASYMPTOTE_PSVIEWER
extern string pdfviewer;   // Environment variable ASYMPTOTE_PDFVIEWER
extern string ghostscript; // Environment variable ASYMPTOTE_GS
  
extern string PSViewer;
extern string PDFViewer;
extern string Ghostscript;
extern string LaTeX;
extern string Dvips;
extern string Convert;
extern string Display;
extern string Animate;
extern string Python;
extern string Xasy;
extern const string docdir;
  
extern string newline;  
  
extern int safe;
enum origin {CENTER,BOTTOM,TOP,ZERO};
//extern int origin;
  
extern int ShipoutNumber;
  
extern const string suffix;
extern const string guisuffix;
  
extern bool TeXinitialized; // Is LaTeX process initialized?
extern string initdir;

extern string paperType;
extern double pageWidth;
extern double pageHeight;
  
extern int scrollLines;
  
types::record *getSettingsModule();

vm::item &getSetting(string name);
  
template <typename T>
inline T getSetting(string name)
{
  return vm::get<T>(getSetting(name));
}

#if INTV
extern int verbose;
#else
extern vm::item *verboseItem;
inline int verbose() {
  return vm::get<int>(*verboseItem);
}
#endif

extern vm::item *debugItem;
inline bool debug() {
  return vm::get<bool>(*debugItem);
}

bool view();
bool trap();

void setOptions(int argc, char *argv[]);

// Access the arguments once options have been parsed.
int numArgs();
char *getArg(int n);
}
#endif
