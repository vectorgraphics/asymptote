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

namespace settings {
  extern const char PROGRAM[];
  extern const char VERSION[];
  extern const char BUGREPORT[];

  extern std::string outformat;
  extern int keep;
  extern int texprocess;
  extern int verbose;
  extern int view;
  extern int safe;
  extern int autoplain;
  extern int parseonly;
  extern int translate;
  extern int trap;
  extern double deconstruct;
  extern int clearGUI;
  extern int ignoreGUI;
  extern camp::pair postscriptOffset;
  extern int bottomOrigin;
  extern int topOrigin;
  
  extern double defaultlinewidth;
  extern double defaultfontsize;
  extern bool suppressOutput;
  extern bool upToDate;
  extern int overwrite;

  extern int ShipoutNumber;
  
  extern const std::string suffix;
  extern const std::string guisuffix;
  
  const char *getAsyDir(); // Returns the environment asymptote directory.

  extern std::string outname; 
  extern std::list<std::string> *outnameStack;
  
  extern bool TeXinitialized; // Is LaTeX process initialized?

  extern char* AsyDir; // Environment variable ASYMPTOTE_DIR
  extern std::string PSViewer; // Environment variable ASYMPTOTE_PSVIEWER
  extern std::string PDFViewer; // Environment variable ASYMPTOTE_PSVIEWER
  extern std::string paperType;
  extern double pageWidth;
  extern double pageHeight;
  
  void setOptions(int argc, char *argv[]);

  // Access the arguments once options have been parsed.
  int numArgs();
  char *getArg(int n);
  
  void reset(); // Reset to startup defaults
}
#endif
