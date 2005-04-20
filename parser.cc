/*****
 * parser.cc
 * Tom Prince 2004/01/10
 *
 *****/

#include <string>
#include <fstream>
#include <sstream>

#include "interact.h"
#include "locate.h"
#include "errormsg.h"
#include "parser.h"

using std::string;

// The lexical analysis and parsing functions used by parseFile.
void setlexer(size_t (*input) (char* bif, size_t max_size),
              string filename);
extern bool yyparse(void);
extern int yydebug;
extern int yy_flex_debug;
static const int YY_NULL = 0;

namespace parser {

namespace yy { // Lexers

std::streambuf *sbuf = NULL;

size_t stream_input(char *buf, size_t max_size)
{
  size_t count= sbuf ? sbuf->sgetn(buf,max_size) : 0;
  return count ? count : YY_NULL;
}

} // namespace yy

void debug(bool state)
{
  // For debugging the lexer and parser that were machine generated.
  yy_flex_debug = yydebug = state;
}

absyntax::file *doParse(size_t (*input) (char* bif, size_t max_size),
                        string filename)
{
  setlexer(input,filename);
  absyntax::file *root = yyparse() == 0 ? absyntax::root : 0;
  yy::sbuf = 0;
  if (!root) {
    em->error(position::nullPos());
    *em << "error: could not load module '" << filename << "'\n";
    em->sync();
    throw handled_error();
  }
  return root;
}

absyntax::file *parseStdin()
{
  yy::sbuf = std::cin.rdbuf();
  return doParse(yy::stream_input,"-");
}

absyntax::file *parseFile(string filename)
{
  if (filename == "-")
    return parseStdin();
  
  string file = settings::locateFile(filename);

  if (file.empty())
    return 0;

  debug(false); 

  std::filebuf filebuf;
  if (!filebuf.open(file.c_str(),std::ios::in))
    return 0;
  // Check that the file can actually be read.
  try {
    filebuf.sgetc();
  } catch (...) {
    return 0;
  }
  yy::sbuf = &filebuf;
  
  return doParse(yy::stream_input,filename);
}

absyntax::file *parseString(string code)
{
  std::stringbuf buf(code);
  yy::sbuf = &buf;
  return doParse(yy::stream_input,"<eval>");
}

absyntax::file *parseInteractive()
{
  debug(false);
  
#if defined(HAVE_LIBREADLINE) && defined(HAVE_LIBCURSES)
  return doParse(interact::interactive_input,"-");
#else
  return parseStdin();
#endif
}

}
