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
#include <iostream>

using std::cout;
using std::cerr;
using std::endl;
using std::ostringstream;

size_t findextension(const std::string& name, const std::string& suffix);
  
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
  
extern "C" int kill(pid_t pid, int sig);
extern "C" char *strsignal(int sig);
extern "C" int snprintf(char *str, size_t size, const  char  *format,...);
extern "C" char *index(const char *s, int c);
extern "C" char *rindex(const char *s, int c);

extern bool False;

// Like strcpy but allow overlap
inline char *Strcpy(char *dest, const char *src) 
{
  size_t len=strlen(src);
  const char *stop=src+len;
  char *q=dest;
  for(const char *p=src; p <= stop; p++) *(q++)=*p;
  return dest;
}

// Like Strcpy but copies in reverse direction
inline char *rStrcpy(char *dest, const char *src) 
{
  size_t len=strlen(src);
  char *p=dest+len;
  for(const char *r=src+len; r >= src; r--) *(p--)=*r;
  return dest;
}

// insert src at location dest; return location after insertion
inline char *insert(char *dest, const char *src) 
{
  size_t len=strlen(src);
  rStrcpy(dest+len,dest);
  for(size_t i=0; i < len; i++) dest[i]=src[i];
  return dest+len;
}

// delete n characters at location dest
inline void remove(char *dest, unsigned int n) 
{
  if(n > 0) Strcpy(dest,dest+n);
}

#endif
