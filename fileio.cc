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

ofile Stdout("");

void ifile::ignoreComment(bool readstring)
{
  if(comment == 0) return;
  int c;
  bool eol=(stream->peek() == '\n');
  if(eol && (readstring || (csvmode && nullfield))) return;
  for(;;) {
    while(isspace(c=stream->peek())) {
      if(c == '\n' && readstring) return;
      stream->ignore();
      whitespace += (char) c;
    }
    if(c == comment) {
      whitespace="";
      while((c=stream->peek()) != '\n' && c != EOF)
	stream->ignore();
      if(c == '\n') {
	if(readstring) return;
	stream->ignore();
      }
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
  
void ifile::Read(string& val)
{
  string s;
  if(csvmode) {
    bool quote=false;
    while(stream->good()) {
      int c=stream->peek();
      if(c == '"') {quote=!quote; stream->ignore(); continue;}
      if(!quote && (c == ',' || c == '\n')) {
	if(c == '\n') ignoreComment(true);
	break;
      }
      s += (char) stream->get();
    }
  } else if(wordmode) {
    s=string(); 
    *stream >> s;
    whitespace="";
  } else
    getline(*stream,s);
  
  if(comment) {
    size_t p=0;
    while((p=s.find(comment,p)) < string::npos) {
      if(p+1 < s.length() && s[p+1] == comment) {
	s.erase(p,1);
	++p;
      } else {
	s.erase(p);	
	break;
      }
    }
    val=whitespace+s;
  }
}
  
void ofile::writeline() 
{
  if(standard && interact::interactive && !vm::indebugger) {
    int scroll=settings::getScroll();
    if(scroll && lines > 0 && lines % scroll == 0) {
      for(;;) {
	if(!cin.good()) {
	  *stream << newline;
	  cin.clear();
	  break;
	}
	int c=cin.get();
	if(c == '\n') break;
	// Discard any additional characters
	while(cin.good() && cin.get() != '\n');
	if(c == 'q') {lines=0; throw quit();}
      }
    } else *stream << newline;
    ++lines;
  } else *stream << newline;
  if(errorstream::interrupt) {lines=0; throw interrupted();}
}
  
} // namespace camp
