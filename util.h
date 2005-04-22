/*****
 * util.h
 * Andy Hammerlindl 2004/05/10
 *
 * A place for useful utility functions.
 *****/

#ifndef UTIL_H
#define UTIL_H

#include <sys/types.h>
#include <string>
#include <strings.h>
#include <iostream>

using std::cout;
using std::cerr;
using std::endl;
using std::ostringstream;

std::string stripext(const std::string& name, const std::string& suffix);
  
// Construct a filename from the original, adding aux at the end, and
// changing the suffix.
std::string buildname(std::string filename, std::string suffix="",
		      std::string aux="");

// Construct an alternate filename for a temporary file.
std::string auxname(std::string filename, std::string suffix="");

bool checkFormatString(const std::string& format);

// Similar to the standard system call except allows interrupts and does
// not invoke a shell.
int System(const char *command, bool quiet=false, bool wait=true,
	   int *pid=NULL, bool warn=true);
int System(const ostringstream& command, bool quiet=false, bool wait=true,
	   int *pid=NULL, bool warn=true); 
  
#ifdef __DECCXX_LIBCXX_RH70
extern "C" int kill(pid_t pid, int sig) throw();
extern "C" char *strsignal(int sig);
extern "C" int snprintf(char *str, size_t size, const  char  *format,...);
extern "C" double asinh(double x);
extern "C" double acosh(double x);
extern "C" double atanh(double x);
extern "C" double cbrt(double x);
extern "C" double remainder(double x, double y);
extern "C" double hypot(double x, double y) throw();
extern "C" double jn(int n, double x);
extern "C" double yn(int n, double x);
#endif

extern bool False;

// Strip blank lines (which would break the bidirectional TeX pipe)
std::string stripblanklines(std::string& s);

extern char *currentpath;

char *startPath();
char *getPath(char *p=currentpath);
int setPath(const char *s);

#endif
