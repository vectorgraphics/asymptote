/*****
 * interact.cc
 *
 * The glue between the lexical analyzer and the readline library.
 *****/

#include <cstdlib>
#include <cassert>
#include <iostream>
#include <sstream>
#include <sys/wait.h>
#include <unistd.h>
#include "interact.h"

#if defined(HAVE_LIBREADLINE) && defined(HAVE_LIBCURSES)
#include <readline/readline.h>
#include <readline/history.h>
#include <csignal>
#endif

#include "util.h"
#include "errormsg.h"

using std::cout;
using namespace settings;

namespace interact {

int interactive=false;
bool virtualEOF=true;
bool resetenv;
bool uptodate=true;

#if defined(HAVE_LIBREADLINE) && defined(HAVE_LIBCURSES)

static string historyname="history";
static string localhistoryname=".asy_history";
  
/* Read a string, and return a pointer to it. Returns NULL on EOF. */
char *rl_gets(void)
{
  static char *line_read=NULL;
  /* If the buffer has already been allocated,
     return the memory to the free pool. */
  if(line_read) {
    free(line_read);
    line_read=NULL;
  }
     
  /* Get a line from the user. */
  while((line_read=readline("> "))) {
    if(*line_read == 0) continue;    
    static int pid=0, status=0;
    static bool restart=true;
    if(strcmp(line_read,"help") == 0 || strcmp(line_read,"help;") == 0) {
      if(pid) restart=(waitpid(pid, &status, WNOHANG) == pid);
      if(restart) {
	ostringstream cmd;
	cmd << PDFViewer << " " << docdir << "/asymptote.pdf";
	status=System(cmd,false,false,"ASYMPTOTE_PDFVIEWER","pdf viewer",
		      &pid);
      }
      continue;
    }
    break;
  }
     
  if(!line_read) cout << endl;
  else {
    if(strcmp(line_read,"q") == 0 || strcmp(line_read,"quit") == 0
       || strcmp(line_read,"quit;") == 0
       || strcmp(line_read,"exit") == 0
       || strcmp(line_read,"exit;") == 0)
      return NULL;
  }
  
  /* If the line has any text in it, save it on the history. */
  if(line_read && *line_read) add_history(line_read);
  
  return line_read;
}

void overflow()
{
  cerr << "warning: buffer overflow, input discarded." << endl;
}

void add_input(char *&dest, const char *src, size_t& size)
{
  
  size_t len=strlen(src);
  if(len == 0) return;
  
  if(len >= size) {overflow(); return;}
  
  strcpy(dest,src);
  // Auto-terminate each line:
  if(dest[len-1] != ';') {dest[len]=';'; len++;}
  
  size -= len;
  dest += len;
}
 
static const char *input="input "; 
static const char *inputexpand="erase(); include ";
static size_t ninput=strlen(input);
static size_t ninputexpand=strlen(inputexpand);
  
int readline_startup_hook()
{
#ifdef MSDOS
  rl_set_key("\\M-[3~",rl_delete,rl_get_keymap());
  rl_set_key("\\M-[2~",rl_overwrite_mode,rl_get_keymap());
#endif    
  return 0;
}

size_t interactive_input(char *buf, size_t max_size)
{
  static int nlines=1000;
  static bool first=true;
  static string historyfile;
  static bool inputmode=false;
    
  assert(max_size > 0);
  size_t size=max_size-1;
  char *to=buf;
    
  if(first) {
    first=false;
    historyfile=localhistory ? localhistoryname : (initdir+historyname);
    
    read_history(historyfile.c_str());
    rl_bind_key('\t',rl_insert); // Turn off tab completion
#ifdef MSDOS
    rl_startup_hook=readline_startup_hook;
#endif    
  }

  if(virtualEOF) return 0;
  
  static char *line;

  if(inputmode) {
    inputmode=false;  
    virtualEOF=true;
    line += ninput;
    strcpy(to,inputexpand);
    to += ninputexpand;
    add_input(to,line,size);
    return to-buf;
  }
  
  if(em->errors())
    em->clear();
  
  ShipoutNumber=0;
  
  if((line=rl_gets())) {
    errorstream::interrupt=false;
    virtualEOF=true;
    
    if(strncmp(line,input,ninput) == 0) {
      inputmode=true;
      resetenv=true;
      return 0;
    }
    
    if(strcmp(line,"reset") == 0 || strcmp(line,"reset;") == 0) {
      resetenv=true;
      return 0;
    }
    
    add_input(to,line,size);
    return to-buf;
  } else {
    stifle_history(nlines);
    write_history(historyfile.c_str());
    return 0;
  }
}

#endif

} // namespace interact
