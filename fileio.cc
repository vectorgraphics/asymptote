/******
 * fileio.cc
 * Tom Prince and John Bowman 2004/08/10
 *
 * Handle input/output
 ******/

#include "fileio.h"
#include "settings.h"

using namespace std;

namespace camp {

string tab="\t";
string newline="\n";

ofile Stdout("");
static string asyinput=".asy_input";  
ofile typeout(asyinput);
ifile typein(asyinput);

void ifile::csv() {
  if(!csvmode || stream->eof()) return;
  std::ios::iostate rdstate=stream->rdstate();
  if(stream->fail()) stream->clear();
  if(stream->peek() == ',') stream->ignore();
  else stream->clear(rdstate);
}
  
std::string ifile::getcsvline() 
{
  std::string s="";
  bool quote=false;
  while(stream->good()) {
    int c=stream->peek();
    if(c == '"') {quote=!quote; stream->ignore(); continue;}
    if(!quote && (c == ',' || c == '\n')) {
      if(c == '\n' && !linemode) stream->ignore();
      return s;
    }
    s += (char) stream->get();
  }
  return s;
}
  
} // namespace camp
