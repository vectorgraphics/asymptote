/*****
 * interact.cc
 *
 * The glue between the lexical analyzer and the readline library.
 *****/

#include <cstdlib>
#include <cassert>
#include <iostream>
#include <cassert>

#include "interact.h"

#if defined(HAVE_LIBREADLINE) && defined(HAVE_LIBCURSES)
#include <readline/readline.h>
#include <readline/history.h>
#include <signal.h>
#endif

namespace interact {

namespace {
/* A static variable for holding the line. */
char *line_read=(char *)NULL;
}

bool virtualEOF=true;
bool rejectline=false;
int interactive=false;

#if defined(HAVE_LIBREADLINE) && defined(HAVE_LIBCURSES)

bool redraw=false;

static int start=0;
static int end=0;
  
// Special version of import command that first does a reset
const char *Import="Import "; 
  
/* Read a string, and return a pointer to it. Returns NULL on EOF. */
char *rl_gets(void)
{
  /* If the buffer has already been allocated,
     return the memory to the free pool. */
  if(line_read) {
    free(line_read);
    line_read=(char *) NULL;
  }
     
  /* Get a line from the user. */
  while((line_read=readline("> "))) {
    if(*line_read == 0) continue;    
    if(strcmp(line_read,"reset") != 0) break;
    start=(rejectline && history_length) ? end-1 : end;
  }
     
  if(rejectline && history_length) remove_history(history_length-1);
  
  if(!line_read) std::cout << std::endl;
  else {
    if(strcmp(line_read,"q") == 0 || strcmp(line_read,"quit") == 0)
      return NULL;
    if(strcmp(line_read,"redraw") == 0) {
      redraw=true;
      *line_read=0;
    }
  }
  
  /* If the line has any text in it, save it on the history. */
  if(line_read && *line_read) add_history(line_read);
  
  if(line_read && strncmp(line_read,Import,strlen(Import)) == 0)
    start=(rejectline && history_length) ? end-1 : end;

  return line_read;
}

void add_input(char *&dest, const char *src, size_t& size)
{
  size_t len=strlen(src)+1;
  if(len > size) {
    std::cerr << "Input buffer overflow;" << std::endl;
    exit(1);
  }
  strcpy(dest,src);
  dest[strlen(dest)]=';'; // Auto-terminate each line
  size -= len;
  
  if(strncmp(src,Import,strlen(Import)) == 0) dest[0]='i';
  
  dest += len;
}

size_t interactive_input(char *buf, size_t max_size)
{
  static int nlines=1000;
  static bool first=true;
  static const char *historyfile=".asy_history";
  if(first) {
    first=false;
    read_history(historyfile);
    rl_bind_key('\t',rl_insert); // Turn off tab completion
    signal(SIGINT,SIG_IGN);
  }

  if(virtualEOF) return 0;

  char *line;
  if((line=rl_gets())) {
    if(start == 0) start=history_length;
    char *to=buf;
    int i=start;
    assert(max_size > 0);
    size_t size=max_size-1;
    
    if(HIST_ENTRY *next=history_get(i++)) {
      if(redraw) redraw=false;
      else add_input(to,"suppressoutput(true)",size);
      while(HIST_ENTRY *p=history_get(i++)) {
	add_input(to,next->line,size);
	next=p;
      }
      if(*line) add_input(to,"suppressoutput(false)",size);
      add_input(to,next->line,size);
      add_input(to,"shipout();\n",size);
    }
    end=i-1;
    
    virtualEOF=true;
    return to-buf;
  } else {
    stifle_history(nlines);
    write_history(historyfile);
    return 0;
  }
}

#endif

} // namespace interact

