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
extern bool virtualEOF;
extern bool resetenv;
extern bool uptodate;

size_t interactive_input(char *buf, size_t max_size);
void init_interactive();
  
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
