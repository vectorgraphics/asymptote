/*****
 * interact.h
 *
 * The glue between the lexical analyzer and the readline library.
 *****/

#ifndef INTERACT_H
#define INTERACT_H

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "memory.h"

namespace interact {

extern int interactive;
extern bool uptodate;

void init_interactive();
  
// Read a line from the input, without any processing.
mem::string simpleline(mem::string prompt);

// Add a line of input to the readline history.
void addToHistory(mem::string line);

// Functions to work with the most recently entered line in the history.
mem::string getLastHistoryLine();
void setLastHistoryLine(mem::string line);

// Remove the line last added to the history.
void deleteLastLine();

// Write out the history of input lines to the history file.
void cleanup_interactive();

// This class is used to set a text completion function for readline.  A class
// is used instead the usual function pointer so that information such as the
// current environment can be coded into the function (mimicking a closure).
class completer : public gc {
public:
  virtual ~completer() {};
  virtual char *operator () (const char *text, int state) = 0;
};

void setCompleter(completer *c);

#define YY_READ_BUF_SIZE YY_BUF_SIZE
  
}

#endif // INTERACT_H
