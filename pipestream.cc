/* Pipestream: A simple C++ interface to UNIX pipes
   Copyright (C) 2005-2014 John C. Bowman,
   with contributions from Mojca Miklavec

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

#if !defined(_WIN32)

#include <iostream>
#include <cstring>
#include <cerrno>
#include <sstream>
#include <csignal>
#include <fcntl.h>

#include "pipestream.h"
#include "common.h"
#include "errormsg.h"
#include "settings.h"
#include "util.h"
#include "interact.h"
#include "lexical.h"
#include "camperror.h"
#include "pen.h"

iopipestream *instance;

void pipeHandler(int)
{
  Signal(SIGPIPE,SIG_DFL);
  instance->pipeclose();
}

void iopipestream::open(const mem::vector<string> &command, const char *hint,
                        const char *application, int out_fileno)
{
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
    kill(0,SIGPIPE);
    _exit(-1);
  }
  instance=this;
  Signal(SIGPIPE,pipeHandler);
  close(out[1]);
  close(in[0]);
  *buffer=0;
  Running=true;
  pipeopen=true;
  pipein=true;
  block(false,true);
}

void iopipestream::eof()
{
  if(pipeopen && pipein) {
    close(in[1]);
    pipein=false;
  }
}

void iopipestream::pipeclose()
{
  if(pipeopen) {
    kill(pid,SIGHUP);
    eof();
    close(out[0]);
    Running=false;
    pipeopen=false;
    waitpid(pid,NULL,0); // Avoid zombies.
  }
}

void iopipestream::block(bool write, bool read)
{
  if(pipeopen) {
    int w=fcntl(in[1],F_GETFL);
    int r=fcntl(out[0],F_GETFL);
    fcntl(in[1],F_SETFL,write ? w & ~O_NONBLOCK : w | O_NONBLOCK);
    fcntl(out[0],F_SETFL,read ? r & ~O_NONBLOCK : r | O_NONBLOCK);
  }
}

ssize_t iopipestream::readbuffer()
{
  ssize_t nc;
  char *p=buffer;
  ssize_t size=BUFSIZE-1;
  errno=0;
  for(;;) {
    if((nc=read(out[0],p,size)) < 0) {
      if(errno == EAGAIN || errno == EINTR) {p[0]=0; break;}
      else {
        ostringstream buf;
        buf << "read from pipe failed: errno=" << errno;
        camp::reportError(buf);
      }
      nc=0;
    }
    p[nc]=0;
    if(nc == 0) {
      if(waitpid(pid,NULL,WNOHANG) == pid)
        Running=false;
      break;
    }
    if(nc > 0) {
      if(settings::verbose > 2) cerr << p;
      break;
    }
  }
  return nc;
}

bool iopipestream::tailequals(const char *buf, size_t len, const char *prompt,
                              size_t plen)
{
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

string iopipestream::readline()
{
  sbuffer.clear();
  int nc;
  do {
    nc=readbuffer();
    sbuffer.append(buffer);
  } while(buffer[nc-1] != '\n' && Running);
  return sbuffer;
}

void iopipestream::wait(const char *prompt)
{
  sbuffer.clear();
  size_t plen=strlen(prompt);

  do {
    readbuffer();
    if(*buffer == 0) camp::reportError(sbuffer);
    sbuffer.append(buffer);

    if(tailequals(sbuffer.c_str(),sbuffer.size(),prompt,plen)) break;
  } while(true);
}

int iopipestream::wait()
{
  for(;;) {
    int status;
    if (waitpid(pid,&status,0) == -1) {
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

void iopipestream::Write(const string &s)
{
  ssize_t size=s.length();
  if(settings::verbose > 2) cerr << s;
  if(write(in[1],s.c_str(),size) != size) {
    camp::reportFatal("write to pipe failed");
  }
}

#endif
