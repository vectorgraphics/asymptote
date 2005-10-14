/******
 * fileio.cc
 * Tom Prince and John Bowman 2004/08/10
 *
 * Handle input/output
 ******/

#include "fileio.h"
#include "settings.h"

namespace camp {

string tab="\t";
string newline="\n";

string asyinput=".asy_input";
  
ofile nullfile("");
ofile Stdout("");
ofile typeout(asyinput);
ifile typein(asyinput);

void ifile::ignoreComment()
{
  if(comment == 0) return;
  int c;
  if(stream->peek() == '\n') return;
  for(;;) {
    while(isspace(c=stream->peek())) {
      stream->ignore();
      whitespace += (char) c;
    }
    if(c == comment) {
      whitespace="";
      while((c=stream->peek()) != '\n' && c != EOF)
	stream->ignore();
      if(c == '\n') stream->ignore();
    } else return;
  }
}
  
bool ifile::eol()
{
  int c;
  while(isspace(c=stream->peek())) {
    stream->ignore();
    if(c == '\n') return true;
    else whitespace += (char) c;
  }
  return false;
}
  
void ifile::csv()
{
  if(!csvmode || stream->eof()) return;
  std::ios::iostate rdstate=stream->rdstate();
  if(stream->fail()) stream->clear();
  int c=stream->peek();
  if(c == ',' || (c == '\n' && !linemode)) stream->ignore();
  else stream->clear(rdstate);
}
  
mem::string ifile::getcsvline() 
{
  string s="";
  bool quote=false;
  while(stream->good()) {
    int c=stream->peek();
    if(c == '"') {quote=!quote; stream->ignore(); continue;}
    if(!quote && (c == ',' || c == '\n')) {
      if(c == '\n') {
	ignoreComment();
	if(!linemode)
	  stream->ignore();
      }
      return s;
    }
    s += (char) stream->get();
  }
  return s;
}
  
} // namespace camp
