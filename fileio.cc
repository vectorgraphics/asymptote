/******
 * fileio.cc
 * Tom Prince 2004/08/10
 *
 * Handle input/output
 ******/

#include "fileio.h"

using namespace std;

namespace camp {

string tab="\t";
string newline="\n";

string file::filename()
{
  return name;
}

class ifile : public file
{
  istream *stream;
  ifstream fstream;
  
public:
  ifile(string name) : file(name) {
    if(name == "") {
      stream=&cin;
      fstream.setstate(std::ios_base::failbit);
    }
    else {
      fstream.open(name.c_str());
      stream=&fstream;
    }
  }
  
  const char* Mode() {return "input";}
  
  void csv() {
    if(!csvmode || stream->eof()) return;
    ios::iostate rdstate=stream->rdstate();
    if(stream->fail()) stream->clear();
    if(stream->peek() == ',') stream->ignore();
    else stream->clear(rdstate);
  }
  
  bool eol() {
    int c;
    while(isspace(c=stream->peek())) {
      stream->ignore();
      if(c == '\n') return true;
    }
    return false;
  }
  
  bool text() {return true;}
  bool eof() {return stream->eof();}
  bool error() {return stream->fail();}
  void close() {if(fstream) {fstream.close(); closed=true;}}
  void clear() {stream->clear();}
  
public:

  string getcsvline() {
    string s="";
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
  
  // Skip over white space
  void readwhite(string& val) {val=string(); *stream >> val; csv();}
  
  void read(bool &val) {string t; readwhite(t); val=(t == "true"); csv();}
  void read(int& val) {val=0; *stream >> val; csv();}
  void read(double& val) {val=0.0; *stream >> val; csv();}
  void read(pair& val) {val=0.0; *stream >> val; csv();}
  void read(char& val) {val=char(); stream->get(val); csv();}
  void read(string& val) {
    val=string();
    if(csvmode) {
      val=getcsvline();
      csv();
    } else getline(*stream,val);
  }

};

ofile Stdout("");

#ifdef HAVE_RPC_RPC_H

class ixfile : public file
{
  xdr::ixstream stream;
public:
  ixfile(string name) : file(name), stream(name.c_str()) {}  

  const char* Mode() {return "xinput";}
  
  bool eof() {return stream.eof();}
  bool error() {return stream.fail();}
  void close() {stream.close(); closed=true;}
  void clear() {stream.clear();}
  
  void read(int& val) {val=0; stream >> val;}
  void read(double& val) {val=0.0; stream >> val;}
  void read(pair& val) {double x=0.0, y=0.0; stream >> x >> y; val=pair(x,y);}
};

class oxfile : public file
{
  xdr::oxstream stream;
public:
  oxfile(string name) : file(name), stream(name.c_str()) {}  

  const char* Mode() {return "xoutput";}
  
  bool eof() {return stream.eof();}
  bool error() {return stream.fail();}
  void close() {stream.close(); closed=true;}
  void clear() {stream.clear();}
  void flush() {stream.flush();}
  
  void write(int val) {stream << val;}
  void write(double val) {stream << val;}
  void write(const pair& val) {stream << val.getx() << val.gety();}
};

#endif

file* file::open(string name, mode mode_)
{
  file *f;
  switch (mode_) {
    case in:
      f=new ifile(name);
      break;
    case out:
      f=new ofile(name);
      break;
      
#ifdef HAVE_RPC_RPC_H
    case xin:
      f=new ixfile(name);
      break;
    case xout:
      f=new oxfile(name);
      break;
#endif
      
    default:
      reportError("Internal error: invalid file mode.");
      return NULL;
  }
  if(f->error()) {
    std::ostringstream buf;
    buf << "Cannot open file \"" << name << "\".";
    reportError(buf.str().c_str());
  }
  return f;
}

} // namespace camp
