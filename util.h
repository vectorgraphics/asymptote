/*****
 * util.h
 * Andy Hammerlindl 2004/05/10
 *
 * A place for useful utility functions.
 *****/

#ifndef UTIL_H
#define UTIL_H

#include <sys/types.h>
#include <iostream>

#include "common.h"

#include <strings.h>

// Duplicate a string.
char *Strdup(string s);
char *StrdupNoGC(string s);
char *StrdupMalloc(string s);
  
// Strip the directory from a filename.
string stripDir(string name);
  
// Strip the file from a filename, returning the directory.
string stripFile(string name);

// Strip the extension from a filename.
string stripExt(string name, const string& suffix="");
  
void writeDisabled();
  
// Check if global writes are disabled and name contains a directory.
void checkLocal(string name);
  
// Construct a filename from the original, adding aux at the end, and
// changing the suffix.
string buildname(string filename, string suffix="",
		      string aux="", bool stripdir=true);

// Construct an alternate filename for a temporary file in the current
// directory.
string auxname(string filename, string suffix="");

// Similar to the standard system call except allows interrupts and does
// not invoke a shell.
int System(const char *command, int quiet=0, bool wait=true,
	   const char *hint=NULL, const char *application="",
	   int *pid=NULL);
int System(const ostringstream& command, int quiet=0, bool wait=true,
	   const char *hint=NULL, const char *application="",
	   int *pid=NULL); 
  
#if defined(__DECCXX_LIBCXX_RH70)
extern "C" int kill(pid_t pid, Int sig) throw();
extern "C" char *strsignal(Int sig);
extern "C" double asinh(double x);
extern "C" double acosh(double x);
extern "C" double atanh(double x);
extern "C" double cbrt(double x);
extern "C" double erf(double x);
extern "C" double erfc(double x);
extern "C" double tgamma(double x);
extern "C" double remainder(double x, double y);
extern "C" double hypot(double x, double y) throw();
extern "C" double jn(Int n, double x);
extern "C" double yn(Int n, double x);
#endif

#if defined(__mips)
extern "C" double tgamma(double x);
#endif

#if defined(__DECCXX_LIBCXX_RH70) || defined(__CYGWIN__)
extern "C" int snprintf(char *str, size_t size, const char *format,...);
extern "C" int fileno(FILE *);
extern "C" char *strptime(const char *s, const char *format, struct tm *tm);
#endif

#if defined(__CYGWIN__)
#define ARG_MAX _POSIX_ARG_MAX
#endif

extern bool False;

// Strip blank lines (which would break the bidirectional TeX pipe)
string stripblanklines(const string& s);

extern char *currentpath;

const char *startPath();
const char* setPath(const char *s, bool quiet=false);
const char *changeDirectory(const char *s);
extern char *startpath;

void backslashToSlash(string& s);
void spaceToUnderscore(string& s);
string Getenv(const char *name, bool msdos);

void execError(const char *command, const char *hint, const char *application);
  
// This invokes a viewer to display the manual.  Subsequent calls will only
// pop-up a new viewer if the old one has been closed.
void popupHelp();

inline Int Abs(Int x) {
#ifdef HAVE_LONG_LONG
  return llabs(x);
#else
#ifdef HAVE_LONG
  return labs(x);
#else
  return abs(x);
#endif
#endif
}

unsigned unsignedcast(Int n);
int intcast(Int n);
  
#endif
