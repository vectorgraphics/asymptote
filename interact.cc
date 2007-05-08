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

using namespace settings;

namespace run {
  void init_readline(bool);
}

namespace interact {

int interactive=false;
bool uptodate=true;

completer *currentCompleter=0;

void setCompleter(completer *c) {
  currentCompleter=c;
}

char *call_completer(const char *text, int state) {
  return currentCompleter ? (*currentCompleter)(text, state) : 0;
}

#if defined(HAVE_LIBREADLINE) && defined(HAVE_LIBCURSES)
void init_completion() {
  rl_completion_entry_function=call_completer;

  rl_completion_append_character='\0'; // Don't add a space after a match.

  // Build a string containing all characters that separate words to be
  // completed.  All characters that can't form part of an identifier are
  // treated as break characters.
  static char break_characters[128];
  int j=0;
  for (unsigned char c=9; c<128; ++c)
    if (!isalnum(c) && c != '_') {
      break_characters[j]=c;
      ++j;
    }
  break_characters[j]='\0';
  rl_completer_word_break_characters=break_characters;
}
#endif  

void pre_readline()
{
#if defined(HAVE_LIBREADLINE) && defined(HAVE_LIBCURSES)
  run::init_readline(getSetting<bool>("tabcompletion"));
#endif  
}

void init_interactive()
{
#if defined(HAVE_LIBREADLINE) && defined(HAVE_LIBCURSES)
  init_completion();
  read_history(historyname.c_str());
#endif  
}
  
#if !defined(HAVE_LIBREADLINE) || !defined(HAVE_LIBCURSES)
char *readline(const char *prompt) {
  if(!cin.good()) {cin.clear(); return NULL;}
  cout << prompt;
  string s;
  getline(cin,s);
  char *p=(char *) malloc(s.size()+1);
  return strcpy(p,s.c_str());
}
#endif  
  
string simpleline(string prompt) {
  // Rebind tab key, as the setting tabcompletion may be changed at runtime.
  pre_readline();

  /* Get a line from the user. */
  char *line = readline(prompt.c_str());

  /* Ignore keyboard interrupts while taking input. */
  errorstream::interrupt=false;

  if (line) {
    string s=line;
    free(line);
    return s;
  } else {
    cout << endl;
    return "\n";
  }
}

void addToHistory(string line) {
#if defined(HAVE_LIBREADLINE) && defined(HAVE_LIBCURSES)
    // Only add it if it has something other than newlines.
    if (line.find_first_not_of('\n') != string::npos) {
      add_history(line.c_str());
    }
#endif    
}

string getLastHistoryLine() {
#if defined(HAVE_LIBREADLINE) && defined(HAVE_LIBCURSES)
  HIST_ENTRY *entry=history_list()[history_length-1];
  if (!entry) {
    em->compiler(position());
    *em << "can't access last history line";
    return "";
  }
  else {
    return entry->line;
  }
#else
  return "";
#endif
}

void setLastHistoryLine(string line) {
#if defined(HAVE_LIBREADLINE) && defined(HAVE_LIBCURSES)
  HIST_ENTRY *entry=remove_history(history_length-1);
  if (!entry) {
    em->compiler(position());
    *em << "can't modify last history line";
  }
  else {
    addToHistory(line);

    free(entry->line);
    free(entry);
  }
#endif
}

void deleteLastLine() {
#if defined(HAVE_LIBREADLINE) && defined(HAVE_LIBCURSES)
  HIST_ENTRY *entry=remove_history(history_length-1);
  if (!entry) {
    em->compiler(position());
    *em << "can't delete last history line";
  }
  else {
    free(entry->line);
    free(entry);
  }
#endif
}

void cleanup_interactive() {
#if defined(HAVE_LIBREADLINE) && defined(HAVE_LIBCURSES)
  // Write the history file.
  stifle_history(getSetting<int>("historylines"));
  write_history(historyname.c_str());
#endif
}

} // namespace interact
