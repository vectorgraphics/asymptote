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

using std::string;

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
  
extern string outformat;
extern int keep;
extern int texprocess;
extern int texmode;
extern int debug;
extern int verbose;
extern int view;
extern int safe;
extern int autoplain;
extern int parseonly;
extern int listvariables;
extern int translate;
extern int bwonly;
extern int grayonly;
extern int rgbonly;
extern int cmykonly;
extern int trap;
extern double deconstruct;
extern int clearGUI;
extern int ignoreGUI;
extern camp::pair postscriptOffset;
enum origin {CENTER,BOTTOM,TOP,ZERO};
extern int origin;
  
extern int ShipoutNumber;
  
extern const string suffix;
extern const string guisuffix;
  
extern string outname; 

extern bool TeXinitialized; // Is LaTeX process initialized?

extern string paperType;
extern double pageWidth;
extern double pageHeight;
  
extern int scrollLines;
  
void setOptions(int argc, char *argv[]);

// Access the arguments once options have been parsed.
int numArgs();
char *getArg(int n);
}
#endif
