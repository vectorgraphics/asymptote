/*****
 * util.cc
 * John Bowman
 *
 * A place for useful utility functions.
 *****/

#ifdef __CYGWIN__
#define _POSIX_C_SOURCE 200809L
#endif

#include <cassert>
#include <iostream>
#include <cstdio>
#include <cfloat>
#include <sstream>
#include <cerrno>
#include <cstdlib>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <cstring>
#include <algorithm>


#if !defined(_WIN32)
#include <sys/wait.h>
#include <sys/param.h>
#include <unistd.h>
#include <dirent.h>
#else
#include <Windows.h>
#include <Shlwapi.h>
#include <direct.h>
#include "win32helpers.h"

#define getcwd _getcwd
#define chdir _chdir
#endif

#include "util.h"
#include "settings.h"
#include "errormsg.h"
#include "camperror.h"
#include "interact.h"
#include "locate.h"

using namespace settings;
using camp::reportError;

bool False=false;

namespace vm {
void error(const char* message);
}

#if __GNUC__
#include <cxxabi.h>
string demangle(const char *s)
{
  int status;
  char *demangled = abi::__cxa_demangle(s,NULL,NULL,&status);
  if (status == 0 && demangled) {
    string str(demangled);
    free(demangled);
    return str;
  } else if (status == -2) {
    free(demangled);
    return s;
  } else {
    free(demangled);
    return string("Unknown(") + s + ")";
  }
}
#else
string demangle(const char* s)
{
  return s;
}
#endif

#pragma
// windows specific unnamed spaces
#if defined(_WIN32)
namespace w32 = camp::w32;

namespace
{
/** @brief System, but for Windows.
 *         Any handle placed in outHandle must be properly closed */
int SystemWin32(const mem::vector<string>& command, int quiet, bool wait,
                const char* hint, const char* application, int* ppid);
}

namespace
{
int SystemWin32(const mem::vector<string>& command, int quiet, bool wait,
                const char* hint, const char* application, int* ppid)
{
  cout.flush();
  if (command.empty())
  {
    camp::reportError("Command cannot be empty");
    return -1;
  }
  string cmdlineStr=camp::w32::buildWindowsCmd(command);

  STARTUPINFOA startInfo={};
  startInfo.cb=sizeof(startInfo);
  startInfo.dwFlags=STARTF_USESTDHANDLES;

  // windows' "/dev/null" file (a.k.a. "NUL")
  HANDLE nulFile=CreateFileA(
          "NUL", GENERIC_WRITE, FILE_SHARE_WRITE | FILE_SHARE_READ,
          nullptr,
          OPEN_EXISTING,
          FILE_ATTRIBUTE_NORMAL,
          nullptr);

  // set quiet info
  startInfo.hStdInput= GetStdHandle(STD_INPUT_HANDLE);
  startInfo.hStdOutput= quiet >= 1 ? nulFile : GetStdHandle(STD_OUTPUT_HANDLE);
  startInfo.hStdOutput= quiet >= 2 ? nulFile : GetStdHandle(STD_ERROR_HANDLE);

  PROCESS_INFORMATION procInfo={};

  w32::checkResult(CreateProcessA(
                           command.at(0).c_str(),
                           cmdlineStr.data(),
                           nullptr, nullptr, false,
                           0,
                           nullptr, nullptr,
                           &startInfo,
                           &procInfo),
                   "Cannot open process");

  w32::checkResult(CloseHandle(nulFile), "Cannot close null file handle");

  if (ppid)
  {
    *ppid=procInfo.dwProcessId;
  }

  w32::HandleRaiiWrapper procHandle(procInfo.hProcess);
  w32::HandleRaiiWrapper threadHandle(procInfo.hThread);


  if (!wait)
  {
    return 0;
  }

  DWORD retcode=-1;
  // else, wait
  switch (WaitForSingleObject(procHandle.getHandle(), INFINITE))
  {
    case WAIT_OBJECT_0: {
      retcode=GetExitCodeProcess(procHandle.getHandle(), &retcode);
      break;
    }
    case WAIT_ABANDONED:// also impossible, we are waiting for a process
    case WAIT_TIMEOUT:  // impossible, since we set timeout to infinite
    case WAIT_FAILED:
    default:
      camp::reportError("Waiting for process failed");
      break;
  }
  return static_cast<int>(retcode);
}
}

#endif

char *Strdup(string s)
{
  size_t size=s.size()+1;
  char *dest=new(UseGC) char[size];
  std::memcpy(dest,s.c_str(),size*sizeof(char));
  return dest;
}

char *StrdupNoGC(string s)
{
  size_t size=s.size()+1;
  char *dest=new char[size];
  std::memcpy(dest,s.c_str(),size*sizeof(char));
  return dest;
}

char *StrdupMalloc(string s)
{
  size_t size=s.size()+1;
  char *dest=(char *) std::malloc(size);
  std::memcpy(dest,s.c_str(),size*sizeof(char));
  return dest;
}

string stripDir(string name)
{
#if defined(_WIN32)
  char* nameBuf=Strdup(std::move(name));
  PathStripPathA(nameBuf);

  return nameBuf;
#else
  size_t p=name.rfind('/');
  if(p < string::npos) name.erase(0,p+1);
  return name;
#endif
}

string stripFile(string name)
{
#if defined(_WIN32)
  std::replace(name.begin(), name.end(), '\\', '/');
#endif

  bool dir=false;
  size_t p=name.rfind('/');
  if(p < string::npos) {
    dir=true;
    while(p > 0 && name[p-1] == '/') --p;
    name.erase(p+1);
  }

  return dir ? name : "";
}

string stripExt(string name, const string& ext)
{
  string suffix="."+ext;
  size_t p=name.rfind(suffix);
  size_t n=suffix.length();
  if(n == 1 || p == name.length()-n)
    return name.substr(0,p);
  else return name;
}

void backslashToSlash(string& s)
{
  size_t p;
  while((p=s.find('\\')) < string::npos)
    s[p]='/';
}

void spaceToUnderscore(string& s)
{
  size_t p;
  while((p=s.find(' ')) < string::npos)
    s[p]='_';
}

string Getenv(const char *name, bool msdos)
{
#if defined(_WIN32)
  size_t envSize=0;
  getenv_s(&envSize,nullptr,0,name);
  if (envSize == 0)
  {
    return "";
  }

  mem::vector<char> resultingData(envSize);
  if (getenv_s(&envSize, resultingData.data(), resultingData.size(), name) != 0)
  {
    camp::reportError("Cannot retrieve environment variable");
  }

  return resultingData.data();

#else
  char *s=getenv(name);
  if(!s) return "";
  string S=string(s);
  if(msdos) backslashToSlash(S);
  return S;
#endif
}

void readDisabled()
{
  camp::reportError("Read from other directories disabled; override with option -globalread");
}

void writeDisabled()
{
  camp::reportError("Write to other directories disabled; override with option -globalwrite");
}

string cleanpath(string name)
{
  string dir=stripFile(name);
  name=stripDir(name);
  spaceToUnderscore(name);
  return dir+name;
}

string inpath(string name)
{
  bool global=globalread();
  string dir=stripFile(name);
  if(global && !dir.empty()) return name;
  string indir=stripFile(outname());
  if(!(global || dir.empty() || dir == indir)) readDisabled();
  return stripDir(name);
}

string outpath(string name)
{
  bool global=globalwrite();
  string dir=stripFile(name);
  if(global && !dir.empty()) return name;
  string outdir=stripFile(outname());
  if(!(global || dir.empty() || dir == outdir)) writeDisabled();
  return outdir+stripDir(name);
}

string buildname(string name, string suffix, string aux)
{
  name=stripExt(outpath(name),defaultformat())+aux;
  if(!suffix.empty()) name += "."+suffix;
  return name;
}

string auxname(string filename, string suffix)
{
  return buildname(filename,suffix,"_");
}

sighandler_t Signal(int signum, sighandler_t handler)
{
#if !defined(_WIN32)
  struct sigaction action,oldaction;
  action.sa_handler=handler;
  sigemptyset(&action.sa_mask);
  action.sa_flags=0;
  return sigaction(signum,&action,&oldaction) == 0 ? oldaction.sa_handler :
    SIG_ERR;
#else
  return signal(signum, handler);
#endif
}

void push_split(mem::vector<string>& a, const string& S)
{
  const char *p=S.c_str();
  string s;
  char c;
  while((c=*(p++))) {
    if(c == ' ') {
      if(s.size() > 0) {
        a.push_back(s);
        s.clear();
      }
    } else s += c;
  }
  if(s.size() > 0)
    a.push_back(s);
}

char **args(const mem::vector<string>& s, bool quiet)
{
  size_t count=s.size();

  char **argv=NULL;
  argv=new char*[count+1];
  for(size_t i=0; i < count; ++i)
    argv[i]=StrdupNoGC(s[i]);

  if(!quiet && settings::verbose > 1) {
    cerr << argv[0];
    for(size_t i=1; i < count; ++i) cerr << " " << argv[i];
    cerr << endl;
  }

  argv[count]=NULL;
  return argv;
}

void execError(const char *command, const char *hint, const char *application)
{
  cerr << "Cannot execute " << command << endl;
  if(*application == 0) application=hint;
  if(hint) {
    string s=string(hint);
    transform(s.begin(), s.end(), s.begin(), toupper);
    cerr << "Please put in a file " << getSetting<string>("config")
         << ": " << endl << endl
         << "import settings;" << endl
         << hint << "=\"LOCATION\";" << endl << endl
         << "where LOCATION specifies the location of "
         << application << "." << endl << endl
         << "Alternatively, set the environment variable ASYMPTOTE_" << s
         << endl << "or use the command line option -" << hint
         << "=\"LOCATION\". For further details, see" << endl
         << "https://asymptote.sourceforge.io/doc/Configuring.html" << endl
         << "https://asymptote.sourceforge.io/doc/Search-paths.html" << endl;
  }
}

// quiet: 0=none; 1=suppress stdout; 2=suppress stdout+stderr.
int System(const mem::vector<string> &command, int quiet, bool wait,
           const char *hint, const char *application, int *ppid)
{
#if _WIN32
    return SystemWin32(command, quiet, wait, hint, application, ppid);
#else
  int status;

  cout.flush(); // Flush stdout to avoid duplicate output.

  char **argv=args(command);

  int pid=fork();
  if(pid == -1)
    camp::reportError("Cannot fork process");

  if(pid == 0) {
    if(interact::interactive) signal(SIGINT,SIG_IGN);
    if(quiet) {
      static int null=creat("/dev/null",O_WRONLY);
      close(STDOUT_FILENO);
      dup2(null,STDOUT_FILENO);
      if(quiet == 2) {
        close(STDERR_FILENO);
        dup2(null,STDERR_FILENO);
      }
    }
    if(argv) {
      execvp(argv[0],argv);
      execError(argv[0],hint,application);
      _exit(-1);
    }
  }

  if(ppid) *ppid=pid;
  for(;;) {
    if(waitpid(pid, &status, wait ? 0 : WNOHANG) == -1) {
      if(errno == ECHILD) return 0;
      if(errno != EINTR) {
        if(quiet < 2) {
          ostringstream msg;
          msg << "Command failed: ";
          for(size_t i=0; i < command.size(); ++i) msg << command[i] << " ";
          camp::reportError(msg);
        }
      }
    } else {
      if(!wait) return 0;
      if(WIFEXITED(status)) {
        if(argv) {
          char **p=argv;
          char *s;
          while((s=*(p++)) != NULL)
            delete [] s;
          delete [] argv;
        }
        return WEXITSTATUS(status);
      } else {
        if(quiet < 2) {
          ostringstream msg;
          msg << "Command exited abnormally: ";
          for(size_t i=0; i < command.size(); ++i) msg << command[i] << " ";
          camp::reportError(msg);
        }
      }
    }
  }
#endif
}

string stripblanklines(const string& s)
{
  string S=string(s);
  bool blank=true;
  const char *t=S.c_str();
  size_t len=S.length();

  for(size_t i=0; i < len; i++) {
    if(t[i] == '\n') {
      if(blank) S[i]=' ';
      else blank=true;
    } else if(t[i] != '\t' && t[i] != ' ') blank=false;
  }
  return S;
}

char *startpath=NULL;

void noPath()
{
  camp::reportError("Cannot get current path");
}

char *getPath(char *p)
{
#ifdef MAXPATHLEN
  static size_t size = MAXPATHLEN;
#else
  static size_t size = 1024;
#endif
  if(!p) p=new(UseGC) char[size];
  if(!p) noPath();
  else while(getcwd(p,size) == NULL) {
      if(errno == ERANGE) {
        size *= 2;
        p=new(UseGC) char[size];
      } else {noPath(); p=NULL;}
    }
  return p;
}

const char *setPath(const char *s, bool quiet)
{
  if(startpath == NULL) startpath=getPath(startpath);
  if(s == NULL || *s == 0) s=startpath;
  int rc=chdir(s);
  if(rc != 0) {
    ostringstream buf;
    buf << "Cannot change to directory '" << s << "'";
    camp::reportError(buf);
  }
  char *p=getPath();
  if(p && (!interact::interactive || quiet) && verbose > 1)
    cout << "cd " << p << endl;
  return p;
}

void push_command(mem::vector<string>& a, const string& s)
{
  a.push_back(s);
#ifdef __MSDOS__
  if(s == "cmd") {
    a.push_back("/c");
    a.push_back("start");
    a.push_back("\"\"");
  }
#endif
}

void popupHelp() {
#if defined(_WIN32)
  mem::vector<string> cmd;
  push_command(cmd, getSetting<string>("pdfviewer"));
  if (auto viewerOpts = getSetting<string>("pdfviewerOptions"); !viewerOpts.empty())
  {
    istringstream viewerOptStream(viewerOpts);
    string tmp;
    while (viewerOptStream >> tmp)
    {
      if (!tmp.empty())
      {
        cmd.push_back(tmp);
      }
    }
  }

  cmd.push_back(docdir+dirsep+"asymptote.pdf");
  System(cmd,0,false,"pdfviewer","your PDF viewer");
#else
  // If the popped-up help is already running, pid stores the pid of the viewer.
  static int pid=0;

  // Status is ignored.
  static int status=0;

  // If the help viewer isn't running (or its last run has termined), launch the
  // viewer again.
  if (pid==0 || (waitpid(pid, &status, WNOHANG) == pid)) {
    mem::vector<string> cmd;
    push_command(cmd,getSetting<string>("pdfviewer"));
    string viewerOptions=getSetting<string>("pdfviewerOptions");
    if(!viewerOptions.empty())
      cmd.push_back(viewerOptions);
    cmd.push_back(docdir+dirsep+"asymptote.pdf");
    status=System(cmd,0,false,"pdfviewer","your PDF viewer",&pid);
  }
#endif
}

const char *intrange="integer argument is outside valid range";
const char *uintrange="integer argument is outside valid unsigned range";

unsigned unsignedcast(Int n)
{
  if(n < 0 || n/2 > INT_MAX)
    vm::error(uintrange);
  return (unsigned) n;
}

unsignedInt unsignedIntcast(Int n)
{
  if(n < 0)
    vm::error(uintrange);
  return (unsignedInt) n;
}

int intcast(Int n)
{
  if(Abs(n) > INT_MAX)
    vm::error(intrange);
  return (int) n;
}

Int Intcast(unsignedInt n)
{
  if(n > (unsignedInt) Int_MAX)
    vm::error(intrange);
  return (Int) n;
}

bool fileExists(string const& path)
{
#if defined(_WIN32)
  return PathFileExistsA(path.c_str());
#else
  return access(path.c_str(), R_OK) == 0;
#endif
}
