/*****
 * parser.cc
 * Tom Prince 2004/01/10
 *
 *****/

#include <fstream>
#include <sstream>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifdef HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif

#include "interact.h"
#include "locate.h"
#include "errormsg.h"
#include "parser.h"

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

namespace {
void error(string filename)
{
  em->sync();
  *em << "error: could not load module '" << filename << "'\n";
  em->sync();
  throw handled_error();
}
}

absyntax::file *doParse(size_t (*input) (char* bif, size_t max_size),
                        string filename)
{
  setlexer(input,filename);
  absyntax::file *root = yyparse() == 0 ? absyntax::root : 0;
  absyntax::root = 0;
  yy::sbuf = 0;
  if(!root) {
    em->error(position::nullPos());
    if(!interact::interactive)
      error(filename);
    else
      throw handled_error();
  }
  return root;
}

absyntax::file *parseStdin()
{
  debug(false);
  yy::sbuf = std::cin.rdbuf();
  return doParse(yy::stream_input,"-");
}

absyntax::file *parseFile(string filename)
{
  if(filename == "-")
    return parseStdin();
  
  string file = settings::locateFile(filename);

  if(file.empty())
    error(filename);

  debug(false); 

  std::filebuf filebuf;
  if(!filebuf.open(file.c_str(),std::ios::in))
    error(filename);
  
#ifdef HAVE_SYS_STAT_H
  // Check that the file is not a directory.
  static struct stat buf;
  if(stat(file.c_str(),&buf) == 0) {
    if(S_ISDIR(buf.st_mode))
      error(filename);
  }
#endif
  
  // Check that the file can actually be read.
  try {
    filebuf.sgetc();
  } catch (...) {
    error(filename);
  }
  
  yy::sbuf = &filebuf;
  return doParse(yy::stream_input,file);
}

absyntax::file *parseString(string code, string filename)
{
  debug(false);
  std::stringbuf buf(code.c_str());
  yy::sbuf = &buf;
  return doParse(yy::stream_input,filename);
}

} // namespace parser

