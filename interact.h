/*****
 * interact.h
 *
 * The glue between the lexical analyzer and the readline library.
 *****/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

namespace interact {

extern int interactive;
extern bool virtualEOF;
extern bool rejectline;

size_t interactive_input(char *buf, size_t max_size);
  
#define YY_READ_BUF_SIZE YY_BUF_SIZE
  
}
