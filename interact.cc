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
#include "util.h"
#include "symbol.h"
#include "genv.h"

#if defined(HAVE_LIBREADLINE) && defined(HAVE_LIBCURSES)
#include <readline/readline.h>
#include <readline/history.h>
#include <signal.h>
#endif

#include "fileio.h"

namespace interact {

using namespace std;  
  
bool virtualEOF=true;
bool rejectline=false;
int interactive=false;

#if defined(HAVE_LIBREADLINE) && defined(HAVE_LIBCURSES)

bool redraw=false;

static int start=0;
static int end=0;
  
static string asyinput=".asy_input";  
static const char *historyfile=".asy_history";
  
void reset()
{
  start=(rejectline && history_length) ? end-1 : end;
  camp::typeout->seek(0);
}
  
static const char *input="input "; 
static size_t ninput=strlen(input);
  
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
    if(strncmp(line_read,input,ninput) == 0) {reset(); break;}
    if(strcmp(line_read,"reset") != 0 && strcmp(line_read,"reset;") != 0)
      break;
    reset();
  }
     
  if(rejectline && history_length) remove_history(history_length-1);
  
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

void overflow() {
  cerr << "warning: buffer overflow, input discarded." << endl;
}

void add_input(char *&dest, const char *src, size_t& size)
{
  if(strncmp(src,input,ninput) == 0) {
    string name(src+ninput);
    size_t p=name.find(';');
    if(p < string::npos) {
      name.erase(p,name.length()-p);
      src++;
    }
    src += name.length()+ninput;
    const string iname=trans::symbolToFile(trans::symbol::trans(name));
    static filebuf filebuf;
    if(!filebuf.open(iname.c_str(),ios::in)) {
      cout << "warning: input file '" << name << "' not found" << endl;
      return;
    }
    size_t len=filebuf.sgetn(dest,size);
    filebuf.close();
    settings::suppressOutput=false;
    if(len == size) {overflow(); return;}
    size -= len;
    dest += len;
  }
  
  size_t len=strlen(src)+1;
  if(len == 1) return;
  
  if(len > size) {overflow(); return;}
  
  strcpy(dest,src);
  dest[strlen(dest)]=';'; // Auto-terminate each line
  
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
    signal(SIGINT,SIG_IGN);
    camp::typeout=new camp::ofile(asyinput);
    camp::typein=new camp::ifile(asyinput);
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
      else {
	add_input(to,"suppressoutput(true)",size);
	camp::typein->seek(0);
      }
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

