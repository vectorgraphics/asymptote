/*****
 * settings.h
 * Andy Hammerlindl 2004/05/10
 *
 * Declares a list of global variables that act as settings in the system.
 *****/

#ifndef SETTINGS_H
#define SETTINGS_H

#include <fstream>

#include "common.h"
#include "pair.h"
#include "item.h"

namespace types {
  class record;
}

namespace settings {
extern const char PROGRAM[];
extern const char VERSION[];
extern const char BUGREPORT[];

extern const string docdir;
  
extern bool safe;
  
extern bool global();

enum origin {CENTER,BOTTOM,TOP,ZERO};
//extern int origin;
  
extern int ShipoutNumber;
  
extern const string suffix;
extern const string guisuffix;
  
extern string historyname;
  
void SetPageDimensions();

types::record *getSettingsModule();

vm::item& Setting(string name);
  
template <typename T>
inline T getSetting(string name)
{
  return vm::get<T>(Setting(name));
}

extern int verbose;
extern bool gray;
extern bool bw;
extern bool rgb;
extern bool cmyk;
  
bool view();
bool trap();
string outname();

void setOptions(int argc, char *argv[]);

// Access the arguments once options have been parsed.
int numArgs();
char *getArg(int n);
 
int getScroll();
  
bool pdf(const string& texengine);
bool latex(const string& texengine);
  
string nativeformat();
string defaultformat();
  
const char *beginlabel(const string& texengine);
const char *endlabel(const string& texengine);
const char *rawpostscript(const string& texengine);
const char *beginspecial(const string& texengine);
const char *endspecial();
  
extern bool fataltex[];
const char **texabort(const string& texengine);
  
string texengine();
string texprogram();
}
#endif
