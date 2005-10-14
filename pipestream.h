/* Pipestream: A simple C++ interface to UNIX pipes
   Version 0.01
   Copyright (C) 2005 John C. Bowman

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA. */

#ifndef PIPESTREAM_H
#define PIPESTREAM_H

#include <iostream>
#include <string>
#include <sys/wait.h>
#include <cerrno>
#include <sstream>
#include <unistd.h>
#include <signal.h>

#include "errormsg.h"
#include "settings.h"
#include "util.h"
#include "interact.h"

using std::string;

char **args(const char *command);
  
// bidirectional stream for reading and writing to pipes
class iopipestream {
protected:
  int in[2];
  int out[2];
  static const int BUFSIZE=SHRT_MAX;
  char buffer[BUFSIZE];
  int pid;
  bool pipeopen;
public:
  void open(const char *command, const char *hint=NULL,
	    const char *application="", int out_fileno=STDOUT_FILENO) {
    if(pipe(in) == -1) {
      cerr << "in pipe failed: " << command << endl;
      exit(-1);
    }

    if(pipe(out) == -1) {
      ostringstream buf;
      cerr << "out pipe failed: " << command << endl;
      exit(-1);
    }
    
    cout.flush(); // Flush stdout to avoid duplicate output.
    
    int wrapperpid;
    
    // Portable way of forking that avoids zombie child processes
    if((wrapperpid=fork()) < 0) {
      cerr << "fork failed: " << command << endl;
      exit(-1);
    }
    
    if(wrapperpid == 0) {
      if((pid=fork()) < 0) {
	cerr << "fork failed: " << command << endl;
	exit(-1);
      }
    
      if(pid == 0) { 
	if(interact::interactive) signal(SIGINT,SIG_IGN);
	close(in[1]);
	close(out[0]);
	dup2(in[0],STDIN_FILENO);
	dup2(out[1],out_fileno);
	close(in[0]);
	close(out[1]);
	char **argv=args(command);
	if(argv) execvp(argv[0],argv);
	execError(command,hint,application);
      }
      exit(0);
    } else {
      close(out[1]);
      close(in[0]);
      pipeopen=true;
      waitpid(wrapperpid,NULL,0);
    }
  }

  iopipestream(): pid(0), pipeopen(false) {}
  
  iopipestream(const char *command, const char *hint=NULL,
	       const char *application="", int out_fileno=STDOUT_FILENO) :
    pid(0), pipeopen(false) {
    open(command,hint,application,out_fileno);
  }
  
  void pipeclose() {
    if(pipeopen) {
      close(in[1]);
//      close(out[0]);
      pipeopen=false;
    }
  }
  
  ~iopipestream() {
    pipeclose();
  }
    
  ssize_t readbuffer() {
    ssize_t nc;
    char *p=buffer;
    ssize_t size=BUFSIZE-1;
    for(;;) {
      if((nc=read(out[0],p,size)) < 0) {
	camp::reportError("read from pipe failed");
      }
      p[nc]=0;
      if(nc == 0) break;
      if(nc > 0) {
	if(settings::verbose > 2) std::cerr << p << std::endl;
	if(strchr(p,'\n')) break;
	p += nc;
	size -= nc;
      }
    }
    return p+nc-buffer;
  }

  typedef iopipestream& (*imanip)(iopipestream&);
  iopipestream& operator << (imanip func) { return (*func)(*this); }
  
  iopipestream& operator >> (string& s) {
    readbuffer();
    s=buffer;
    return *this;
  }
  
  void wait(const char *prompt, const char *abort=NULL) {
    ssize_t len;
    size_t plen=strlen(prompt);
    size_t alen=0;
    if(abort) alen=strlen(abort);
    do {
      len=readbuffer();
      if(abort) {
	if(strncmp(buffer,abort,alen) == 0) {
	  camp::TeXcontaminated=true;
	  if(settings::texmode) return;
	  camp::reportError(buffer);
	}
	char *p=buffer;
	while((p=strchr(p,'\n')) != NULL) {
	  ++p;
	  if(strncmp(p,abort,alen) == 0)
	    camp::reportError(buffer);
	}
      }
    } while (strcmp(buffer+len-plen,prompt) != 0);
  }

  int wait() {
    for(;;) {
      int status;
      if (waitpid(pid, &status, 0) == -1) {
	if (errno == ECHILD) return 0;
	if (errno != EINTR) {
	  ostringstream buf;
	  buf << "Process " << pid << " failed";
	  camp::reportError(buf);
	}
      } else {
	if(WIFEXITED(status)) return WEXITSTATUS(status);
	else {
	  ostringstream buf;
	  buf << "Process " << pid << " exited abnormally";
	  camp::reportError(buf);
	}
      }
    }
  }
  
  iopipestream& operator << (const string &s) {
    ssize_t size=s.length();
    if(settings::verbose > 2) std::cerr << s << std::endl;
    if(write(in[1],s.c_str(),size) != size)
      camp::reportError("write to pipe failed");
    return *this;
  }
  
  template<class T>
  iopipestream& operator << (T x) {
    std::ostringstream os;
    os << x;
    *this << os.str();
    return *this;
  }
};
  
#endif

