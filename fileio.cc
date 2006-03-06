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

void ifile::ignoreComment(bool readstring)
{
  if(comment == 0) return;
  int c;
  bool eol=(stream->peek() == '\n');
  if(eol && (readstring || (csvmode && nullfield))) return;
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
    } else {if(eol) stream->unget(); return;}
  }
}
  
bool ifile::eol()
{
  int c;
  while(isspace(c=stream->peek())) {
    if(c == '\n') return true;
    else {
      stream->ignore();
      whitespace += (char) c;
    }
  }
  return false;
}
  
bool ifile::nexteol()
{
  int c;
  if(nullfield) {
    nullfield=false;
    return true;
  }
  
  while(isspace(c=stream->peek())) {
    if(c == '\n' && comma) {
      nullfield=true;
      return false;
    }
    stream->ignore();
    if(c == '\n') {
      while(isspace(c=stream->peek())) {
	if(c == '\n') {nullfield=true; return true;}
        else {
	  stream->ignore();
	  whitespace += (char) c;
	}
      }
      return true;
    }
    else whitespace += (char) c;
  }
  return false;
}
  
void ifile::csv()
{
  comma=false;
  nullfield=false;
  if(!csvmode || stream->eof()) return;
  std::ios::iostate rdstate=stream->rdstate();
  if(stream->fail()) stream->clear();
  int c=stream->peek();
  if(c == ',' || (c == '\n' && !linemode)) stream->ignore();
  else stream->clear(rdstate);
  if(c == ',') comma=true;
}
  
mem::string ifile::getcsvline() 
{
  string s="";
  bool quote=false;
  while(stream->good()) {
    int c=stream->peek();
    if(c == '"') {quote=!quote; stream->ignore(); continue;}
    if(!quote && (c == ',' || c == '\n')) {
      if(c == '\n') ignoreComment(true);
      return s;
    }
    s += (char) stream->get();
  }
  return s;
}
  
} // namespace camp
