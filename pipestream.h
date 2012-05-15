/* Pipestream: A simple C++ interface to UNIX pipes
   Version 0.04
   Copyright (C) 2005-2009 John C. Bowman

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA. */

#ifndef PIPESTREAM_H
#define PIPESTREAM_H

#include <iostream>
#include <cstring>
#include <sys/wait.h>
#include <cerrno>
#include <sstream>
#include <unistd.h>
#include <signal.h>
#include <fcntl.h>

#include "common.h"
#include "errormsg.h"
#include "settings.h"
#include "util.h"
#include "interact.h"

// bidirectional stream for reading and writing to pipes
class iopipestream {
protected:
  int in[2];
  int out[2];
  static const int BUFSIZE=SHRT_MAX;
  char buffer[BUFSIZE];
  int pid;
  bool Running;
  bool pipeopen;
  bool pipein;
  ostringstream transcript;
public:
  
  void clear() {
    transcript << buffer;
    *buffer=0;
  }
  
  void shred() {
    transcript.str("");
  }
  
  const string message() const {
    return transcript.str();
  }
  
  void open(const mem::vector<string> &command, const char *hint=NULL,
            const char *application="", int out_fileno=STDOUT_FILENO) {
    if(pipe(in) == -1) {
      ostringstream buf;
      buf << "in pipe failed: ";
      for(size_t i=0; i < command.size(); ++i) buf << command[i];
      camp::reportError(buf);
    }

    if(pipe(out) == -1) {
      ostringstream buf;
      buf << "out pipe failed: ";
      for(size_t i=0; i < command.size(); ++i) buf << command[i];
      camp::reportError(buf);
    }
    cout.flush(); // Flush stdout to avoid duplicate output.
    
    if((pid=fork()) < 0) {
      ostringstream buf;
      buf << "fork failed: ";
      for(size_t i=0; i < command.size(); ++i) buf << command[i];
      camp::reportError(buf);
    }
    
    if(pid == 0) { 
      if(interact::interactive) signal(SIGINT,SIG_IGN);
      close(in[1]);
      close(out[0]);
      close(STDIN_FILENO);
      close(out_fileno);
      dup2(in[0],STDIN_FILENO);
      dup2(out[1],out_fileno);
      close(in[0]);
      close(out[1]);
      char **argv=args(command);
      if(argv) execvp(argv[0],argv);
      execError(argv[0],hint,application);
      kill(0,SIGTERM);
      _exit(-1);
    }
    close(out[1]);
    close(in[0]);
    *buffer=0;
    pipeopen=true;
    pipein=true;
    Running=true;
  }

  void block(bool block=true) {
    if(pipeopen) {
      int flags=fcntl(out[0],F_GETFL);
      if(block)
        fcntl(out[0],F_SETFL,flags & ~O_NONBLOCK);
      else
        fcntl(out[0],F_SETFL,flags | O_NONBLOCK);
    }
  }
  
  bool isopen() {return pipeopen;}
  
  iopipestream(): pid(0), pipeopen(false) {}
  
  iopipestream(const mem::vector<string> &command, const char *hint=NULL,
               const char *application="", int out_fileno=STDOUT_FILENO) :
    pid(0), pipeopen(false) {
    open(command,hint,application,out_fileno);
  }
  
  void eof() {
    if(pipeopen && pipein) {
      close(in[1]);
      pipein=false;
    }
  }
  
  virtual void pipeclose() {
    if(pipeopen) {
      kill(pid,SIGTERM);
      eof();
      close(out[0]);
      Running=false;
      pipeopen=false;
      waitpid(pid,NULL,0); // Avoid zombies.
    }
  }
  
  virtual ~iopipestream() {
    pipeclose();
  }
    
  ssize_t readbuffer() {
    ssize_t nc;
    char *p=buffer;
    ssize_t size=BUFSIZE-1;
    for(;;) {
      if((nc=read(out[0],p,size)) < 0) {
        if(errno == EAGAIN) {p[0]=0; break;}
        else camp::reportError("read from pipe failed");
      }
      p[nc]=0;
      if(nc == 0) {
        if(waitpid(pid, NULL, WNOHANG) == pid)
          Running=false;
        break;
      }
      if(nc > 0) {
        if(settings::verbose > 2) cerr << p;
        if(strchr(p,'\n')) break;
        p += nc;
        size -= nc;
      }
    }
    return p+nc-buffer;
  }

  bool running() {return Running;}
  
  typedef iopipestream& (*imanip)(iopipestream&);
  iopipestream& operator << (imanip func) { return (*func)(*this); }
  
  iopipestream& operator >> (string& s) {
    readbuffer();
    s=buffer;
    return *this;
  }
  
  bool tailequals(const char *buf, size_t len, const char *prompt,
                  size_t plen) {
    const char *a=buf+len;
    const char *b=prompt+plen;
    while(b >= prompt) {
      if(a < buf) return false;
      if(*a != *b) return false;
      // Handle MSDOS incompatibility:
      if(a > buf && *a == '\n' && *(a-1) == '\r') --a;
      --a; --b;
    }
    return true;
  }
  
  bool checkabort(const char *abort) {
    size_t alen=strlen(abort);
    if(strncmp(buffer,abort,alen) == 0) {
      clear();
      return true;
    }
    char *p=buffer;
    while((p=strchr(p,'\n')) != NULL) {
      ++p;
      if(strncmp(p,abort,alen) == 0) {
        clear();
        return true;
      }
    }
    return false;
  }
  
  // returns true if prompt found, false if abort string is received
  int wait(const char *prompt, const char**abort=NULL) {
    ssize_t len;
    size_t plen=strlen(prompt);
  
    unsigned int n=0;
    if(abort) {
      for(;;) {
        if(abort[n]) n++;
        else break;
      }
    }
    
    do {
      len=readbuffer();
      for(unsigned int i=0; i < n; ++i) 
        if(checkabort(abort[i])) return i+1;
    } while (!tailequals(buffer,len,prompt,plen));
    return 0;
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
  
  void Write(const string &s) {
    ssize_t size=s.length();
    if(settings::verbose > 2) cerr << s;
    if(write(in[1],s.c_str(),size) != size) {
      camp::reportFatal("write to pipe failed");
    }
    
  }
  
  iopipestream& operator << (const string& s) {
    Write(s);
    return *this;
  }
  
  template<class T>
  iopipestream& operator << (T x) {
    ostringstream os;
    os << x;
    Write(os.str());
    return *this;
  }
};
  
#endif
