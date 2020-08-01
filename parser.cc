/*****
 * parser.cc
 * Tom Prince 2004/01/10
 *
 *****/

#include <fstream>
#include <sstream>
#include <cstring>
#include <fcntl.h>
#include <algorithm>

#include "common.h"

#ifdef HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif

#ifdef HAVE_LIBCURL
#include <curl/curl.h>
#endif

#include "interact.h"
#include "locate.h"
#include "errormsg.h"
#include "parser.h"
#include "util.h"

// The lexical analysis and parsing functions used by parseFile.
void setlexer(size_t (*input) (char* bif, size_t max_size), string filename);
extern bool yyparse(void);
extern int yydebug;
extern int yy_flex_debug;
extern bool lexerEOF();
extern void reportEOF();
extern bool hangup;

namespace parser {

namespace yy { // Lexers

std::streambuf *sbuf = NULL;

size_t stream_input(char *buf, size_t max_size)
{
  return sbuf ? sbuf->sgetn(buf,max_size) : 0;
}

} // namespace yy

void debug(bool state)
{
  // For debugging the machine-generated lexer and parser.
  yy_flex_debug = yydebug = state;
}

namespace {
void error(const string& filename)
{
  em.sync();
  em << "error: could not load module '" << filename << "'\n";
  em.sync();
  throw handled_error();
}
}

absyntax::file *doParse(size_t (*input) (char* bif, size_t max_size),
                        const string& filename, bool extendable=false)
{
  setlexer(input,filename);
  absyntax::file *root = yyparse() == 0 ? absyntax::root : 0;
  absyntax::root = 0;
  yy::sbuf = 0;

  if (!root) {
    if (lexerEOF()) {
      if (extendable) {
        return 0;
      } else {
        // Have the lexer report the error.
        reportEOF();
      }
    }

    em.error(nullPos);
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
  yy::sbuf = cin.rdbuf();
  return doParse(yy::stream_input,"-");
}

absyntax::file *parseFile(const string& filename,
                          const char *nameOfAction)
{
  if(filename == "-")
    return parseStdin();
  
#ifdef HAVE_LIBCURL
  string s(filename);
  s.erase(remove_if(s.begin(),s.end(),isspace),s.end());
  if(s.find("http://",0) == 0 || s.find("https://",0) == 0)
    return parseURL(s,nameOfAction);
#endif

  string file = settings::locateFile(filename);

  if(file.empty())
    error(filename);

  if(nameOfAction && settings::verbose > 1)
    cerr << nameOfAction << " " <<  filename << " from " << file << endl;
  
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

absyntax::file *parseString(const string& code,
                            const string& filename,
                            bool extendable)
{
  debug(false);
  stringbuf buf(code);
  yy::sbuf = &buf;
  return doParse(yy::stream_input,filename,extendable);
}

#ifdef HAVE_LIBCURL
size_t WriteMemoryCallback(char *data, size_t size, size_t n, std::string *buf)
{
  size_t Size=size*n;
  buf->append(data,Size);
  return Size;
}

absyntax::file *parseURL(const string& filename,
                         const char *nameOfAction)
{
  string code;

  CURL *curl=curl_easy_init();
  curl_easy_setopt(curl,CURLOPT_URL,filename.c_str());
  curl_easy_setopt(curl,CURLOPT_WRITEFUNCTION,WriteMemoryCallback);
  curl_easy_setopt(curl,CURLOPT_WRITEDATA,&code);
  CURLcode res=curl_easy_perform(curl);

  if(res != CURLE_OK || code == "404: Not Found")
    error(filename);

  if(nameOfAction && settings::verbose > 1)
    cerr << nameOfAction << " " <<  filename << endl;

  curl_easy_cleanup(curl);

  return parseString(code,filename,false);
}
#endif

} // namespace parser

