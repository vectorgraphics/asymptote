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
  
extern bool TeXinitialized; // Is LaTeX process initialized?
extern mem::string historyname;
  
void SetPageDimensions();

types::record *getSettingsModule();

vm::item& Setting(string name);
  
template <typename T>
inline T getSetting(string name)
{
  return vm::get<T>(Setting(name));
}

extern int verbose;
extern string gvOptionPrefix;

bool view();
bool trap();
mem::string outname();

void setOptions(int argc, char *argv[]);

// Access the arguments once options have been parsed.
int numArgs();
char *getArg(int n);
 
int getScroll();
  
bool pdf(const mem::string& texengine);
bool latex(const mem::string& texengine);
  
string nativeformat();
string defaultformat();
  
const char *beginlabel(const mem::string& texengine);
const char *endlabel(const mem::string& texengine);
const char *clip(const mem::string& texengine);
const char *beginspecial(const mem::string& texengine);
const char *endspecial();
const char **texabort(const mem::string& texengine);
  
mem::string texengine();
}
#endif
