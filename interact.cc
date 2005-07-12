/*****
 * interact.cc
 *
 * The glue between the lexical analyzer and the readline library.
 *****/

#include <cstdlib>
#include <cassert>
#include <iostream>
#include "interact.h"

#if defined(HAVE_LIBREADLINE) && defined(HAVE_LIBCURSES)
#include <readline/readline.h>
#include <readline/history.h>
#include <csignal>
#endif

#include "symbol.h"
#include "locate.h"

#include "fileio.h"

using std::cout;

namespace interact {

bool virtualEOF=true;
bool rejectline=false;
int interactive=false;
bool rejectline_cached=false;
  
#if defined(HAVE_LIBREADLINE) && defined(HAVE_LIBCURSES)

bool redraw=false;

static int start=0;
static int end=0;
  
static const char *historyfile=".asy_history";
  
void reset()
{
  start=(rejectline_cached && history_length) ? end-1 : end;
  camp::typeout.seek(0);
}
  
static const char *input="input "; 
static size_t ninput=strlen(input);
  
/* Read a string, and return a pointer to it. Returns NULL on EOF. */
char *rl_gets(void)
{
  static char *line_read=NULL;
  static bool needreset=false;
  
  /* If the buffer has already been allocated,
     return the memory to the free pool. */
  if(line_read) {
    free(line_read);
    line_read=NULL;
  }
     
  /* Get a line from the user. */
  while((line_read=readline("> "))) {
    if(*line_read == 0) continue;    
    if(strncmp(line_read,input,ninput) == 0) {
      if(needreset) reset();
      else needreset=true;
      break;
    }
    if(strcmp(line_read,"reset") != 0 && strcmp(line_read,"reset;") != 0)
      break;
    reset();
    needreset=false;
  }
     
#ifdef HAVE_REMOVE_HISTORY
  if(rejectline_cached && history_length) remove_history(history_length-1);
#endif  
  rejectline_cached=false;
  
  if(!line_read) cout << endl;
  else {
    if(strcmp(line_read,"q") == 0 || strcmp(line_read,"quit") == 0
       || strcmp(line_read,"quit;") == 0)
      return NULL;
    if(strcmp(line_read,"redraw") == 0 || strcmp(line_read,"redraw;") == 0) {
      redraw=true;
      *line_read=0;
    }
  }
  
  /* If the line has any text in it, save it on the history. */
  if(line_read && *line_read) add_history(line_read);
  
  return line_read;
}

void overflow()
{
  cerr << "warning: buffer overflow, input discarded." << endl;
}

void readerror(const string& name) 
{
  cerr << "error: could not load module '" << name << "'" << endl; 
}
  
void add_input(char *&dest, const char *src, size_t& size, bool warn=false)
{
  if(strncmp(src,input,ninput) == 0) {
    string name(src+ninput);
    size_t p=name.find(';');
    if(p < string::npos) {
      name.erase(p,name.length()-p);
      src++;
    }
    src += name.length()+ninput;
    const string iname=settings::locateFile(name);
    std::filebuf filebuf;
    if(!filebuf.open(iname.c_str(),std::ios::in)) {
      if(warn) readerror(name);
      return;
    }
    // Check that the file can actually be read.
    try {
      filebuf.sgetc();
    } catch (...) {
      if(warn) readerror(name);
      return;
    }

    size_t len=filebuf.sgetn(dest,size);
    filebuf.close();
    if(len == size) {overflow(); return;}
    size -= len;
    dest += len;
  }
  
  size_t len=strlen(src);
  if(len == 0) return;
  
  if(len >= size) {overflow(); return;}
  
  strcpy(dest,src);
  // Auto-terminate each line:
  if(dest[len-1] != ';') {dest[len]=';'; len++;}
  
  size -= len;
  dest += len;
}
 
size_t interactive_input(char *buf, size_t max_size)
{
  static int nlines=1000;
  static bool first=true;
  if(first) {
    first=false;
    read_history(historyfile);
    rl_bind_key('\t',rl_insert); // Turn off tab completion
    camp::typeout.open();
    camp::typein.open();
  }

  if(virtualEOF) return 0;
  
  if(rejectline) {
    rejectline_cached=virtualEOF=true;
    return 0;
  }
  
  char *line;

  if((line=rl_gets())) {
    errorstream::interrupt=false;
    if(start == 0) start=history_length;
    char *to=buf;
    int i=start;
    assert(max_size > 0);
    size_t size=max_size-1;
    
    if(HIST_ENTRY *next=history_get(i++)) {
      if(redraw) redraw=false;
      else {
	// Disable stdin/stdout and shipout
	add_input(to,"static {interact(false);}; interact(false)",size);
	camp::typein.seek(0);
      }
      while(HIST_ENTRY *p=history_get(i++)) {
	add_input(to,next->line,size);
	next=p;
      }
      if(*line) // Renable I/O and shipout
	add_input(to,"static {interact(true);}; interact(true);\n",size);
      add_input(to,next->line,size,true);
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

