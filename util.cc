/*****
 * util.cc
 * John Bowman
 *
 * A place for useful utility functions.
 *****/

#include <cassert>
#include <iostream>
#include <cstdio>
#include <string>
#include <cfloat>
#include <sstream>
#include <cerrno>
#include <sys/wait.h>
#include <sys/param.h>
#include <unistd.h>

#include "util.h"
#include "settings.h"
#include "errormsg.h"
#include "camperror.h"
#include "interact.h"

using namespace std;
using namespace settings;

bool False=false;

size_t findextension(const string& name, const string& suffix) 
{
  size_t p=name.rfind("."+suffix);
  if (p == name.length()-suffix.length()-1) return p;
  else return string::npos;
}

string buildname(string filename, string suffix, string aux) 
{
  string name=filename;
  size_t p=name.rfind('/');
  if(p < string::npos) name.erase(0,p+1);
  p=findextension(name,outformat);
  if(p < string::npos) name.erase(p);

  name += aux;
  if(suffix != "") name += "."+suffix;
  return name;
}

string auxname(string filename, string suffix)
{
  return buildname(filename,suffix,"_");
}
  
bool checkFormatString(const string& format)
{
  if(format.find(' ') != string::npos) { // Avoid potential security hole
    ostringstream msg;
    msg << "output format \'" << format << "\' is invalid";
    camp::reportError(msg.str());
    return false;
  }
  return true;
}
  
char **args(const char *command)
{
  if(command == NULL) return NULL;
  char *cmd=strcpy(new char[strlen(command)+1],command);
  char *p=cmd;
  int n=1;
  while((p=index(p,' '))) {n++; p++; while(*p == ' ') p++;}
  char **argv=new char*[n+1];
  argv[0]=p=cmd;
  n=1;
  while((p=index(p,' '))) {*(p++)=0; while(*p == ' ') p++; argv[n++]=p;}
  argv[n]=NULL;
  return argv;
}

int System(const char *command, bool quiet, bool wait, int *ppid, bool warn)
{
  int status;

  if (!command) return 1;
  if(verbose > 1) cerr << command << endl;

  int pid = fork();
  if (pid == -1) {
    camp::reportError("Cannot fork process");
    return 1;
  }
  char **argv=args(command);
  if (pid == 0) {
    if(interact::interactive) signal(SIGINT,SIG_IGN);
    if(quiet) close(STDOUT_FILENO);
    if(argv) execvp(argv[0],argv);
    ostringstream msg;
    if(warn) {
      msg <<  "Cannot execute " << argv[0];
      camp::reportError(msg.str());
    }
    return -1;
  }

  if(ppid) *ppid=pid;
  if(!wait) return 0;
  for(;;) {
    if (waitpid(pid, &status, 0) == -1) {
      if (errno == ECHILD) return 0;
      if (errno != EINTR) {
        ostringstream msg;
        msg << "Command " << command << " failed";
        camp::reportError(msg.str());
        return 1;
      }
    } else {
      if(WIFEXITED(status)) {
	delete [] argv[0];
	delete [] argv;
	return WEXITSTATUS(status);
      }
      else {
        ostringstream msg;
        msg << "Command " << command << " exited abnormally";
        camp::reportError(msg.str());
        return 1;
      }
    }
  }
}

int System(const ostringstream& command, bool quiet, bool wait, int *pid,
	   bool warn) 
{
  return System(command.str().c_str(),quiet,wait,pid,warn);
}

string stripblanklines(string& s)
{
  bool blank=true;
  const char *t=s.c_str();
  size_t len=s.length();
  
  for(size_t i=0; i < len; i++) {
    if(t[i] == '\n') {
      if(blank) s[i]=' ';
      else blank=true;
    } else if(t[i] != '\t' && t[i] != ' ') blank=false;
  }
  return s;
}

static char *startpath=NULL;
char *currentpath=NULL;

char *startPath()
{
  return startpath;
}

void noPath()
{
  camp::reportError("Cannot get current path");
}

char *getPath(char *p)
{
  static int size=MAXPATHLEN;
  if(!p) p=new char[size];
  if(!p) noPath();
  else while(getcwd(p,size) == NULL) {
    if(errno == ERANGE) {
      size *= 2;
      delete [] p;
      p=new char[size];
    } else {noPath(); p=NULL;}
  }
  return p;
}

int setPath(const char *s)
{
  if(s != NULL && *s != 0) {
    if(startpath == NULL) startpath=getPath(startpath);
    return chdir(s);
  } return 0;
}
