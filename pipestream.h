/* Pipestream: A simple C++ interface to UNIX pipes
   Version 0.00
   Copyright (C) 2004 John C. Bowman

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

#include "errormsg.h"
#include "settings.h"
#include "util.h"

char **args(const char *command);
  
extern "C" char *index(const char *s, int c);

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
  void open(const char *command, int out_fileno=STDOUT_FILENO) {
    if(pipe(in) == -1) {
      std::cerr << "in pipe failed: " << command << std::endl;
      throw handled_error();
    }

    if(pipe(out) == -1) {
      std::cerr << "out pipe failed: " << command << std::endl;
      throw handled_error();
    }
    
    if((pid=fork()) < 0) {
      std::cerr << "fork failed: " << command << std::endl;
      throw handled_error();
    }
    
    if(pid == 0) { 
      close(in[1]);
      close(out[0]);
      dup2(in[0],STDIN_FILENO);
      dup2(out[1],out_fileno);
      close(in[0]);
      close(out[1]);
      char **argv=args(command);
      if(argv) execvp(argv[0],argv);
      std::cerr << "exec failed: " << command << std::endl;
      throw handled_error();
    }
    
    close(out[1]);
    close(in[0]);
    pipeopen=true;
  }
  
  iopipestream(): pid(0), pipeopen(false) {}
  
  iopipestream(const char *command, int out_fileno=STDOUT_FILENO) :
    pid(0), pipeopen(false) {open(command,out_fileno);}
  
  iopipestream(const std::string command, int out_fileno=STDOUT_FILENO) :
    pid(0), pipeopen(false) {open(command.c_str(),out_fileno);}
  
  iopipestream(const std::ostringstream& command, int out_fileno=STDOUT_FILENO) :
    pid(0), pipeopen(false) {open(command.str().c_str(),out_fileno);}
  
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
	std::cerr << "read from pipe failed" << std::endl;
	throw handled_error();
      }
      p[nc]=0;
      if(nc == 0) break;
      if(nc > 0) {
	if(settings::verbose > 2) std::cerr << p << std::endl;
	if(index(p,'\n')) break;
	p += nc;
	size -= nc;
      }
    }
    return p+nc-buffer;
  }

  typedef iopipestream& (*imanip)(iopipestream&);
  iopipestream& operator << (imanip func) { return (*func)(*this); }
  
  iopipestream& operator >> (std::string& s) {
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
      if(abort && strncmp(buffer,abort,alen) == 0) {
	std::cerr << buffer << std::endl;
	throw handled_error();
      }
    } while (strcmp(buffer+len-plen,prompt) != 0);
  }

  int wait() {
    for(;;) {
      int status;
      if (waitpid(pid, &status, 0) == -1) {
	if (errno == ECHILD) return 0;
	if (errno != EINTR) {
	  std::cerr << "Process " << pid << " failed" << std::endl;
	  throw handled_error();
	}
      } else {
	if(WIFEXITED(status)) return WEXITSTATUS(status);
	else {
	  std::cerr << "Process " << pid << " exited abnormally" << std::endl;
	  throw handled_error();
	}
      }
    }
  }
  
  iopipestream& operator << (const std::string &s) {
    ssize_t size=s.length();
    if(settings::verbose > 2) std::cerr << s << std::endl;
    if(write(in[1],s.c_str(),size) != size) {
      std::cerr << "write to pipe failed" << std::endl;
      throw handled_error();
    }
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

