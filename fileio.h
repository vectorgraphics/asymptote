/******
 * fileio.h
 * Tom Prince and John Bowman 2004/05/10
 *
 * Handle input/output
 ******/

#ifndef FILEIO_H
#define FILEIO_H

#include <fstream>
#include <iostream>
#include <sstream>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifdef HAVE_RPC_RPC_H
#include "xstream.h"
#endif

#include "pair.h"
#include "triple.h"
#include "guide.h"
#include "pen.h"

#include "camperror.h"
#include "interact.h"
#include "errormsg.h"
#include "memory.h"

namespace camp {

extern string tab;
extern string newline;
  
class file : public gc_cleanup {
protected:  
  string name;
  int nx,ny,nz;    // Array dimensions
  bool linemode;   // Array reads will stop at eol instead of eof.
  bool csvmode;    // Read comma-separated values.
  bool singlemode; // Read/write single-precision XDR values.
  bool closed;     // File has been closed.
  bool checkappend;// Check input for errors/append to output.
  bool standard;   // Standard input/output
  int lines;       // Number of scrolled lines
public: 

  void resetlines() {lines=0;}
  
  bool Standard() {return standard;}
  
  void dimension(int Nx=-1, int Ny=-1, int Nz=-1) {nx=Nx; ny=Ny; nz=Nz;}
  
  file(const string& name, bool checkappend=true) : 
    name(name), linemode(false), csvmode(false), singlemode(false),
    closed(false), checkappend(checkappend), standard(name.empty()),
    lines(0) {dimension();}
  
  virtual void open() {}
  
  void Check() {
    if(error()) {
      std::ostringstream buf;
      buf << "Cannot open file \"" << name << "\".";
      reportError(buf);
    }
  }
  
  virtual ~file() {}

  virtual const char* Mode()=0;

  bool isOpen() {
    if(closed) {
      std::ostringstream buf;
      buf << "I/O operation attempted on closed file \'" << name << "\'.";
      reportError(buf);
    }
    return true;
  }
		
  string filename() {return name;}
  virtual bool eol() {return false;}
  virtual bool nexteol() {return false;}
  virtual bool text() {return false;}
  virtual bool eof()=0;
  virtual bool error()=0;
  virtual void close()=0;
  virtual void clear()=0;
  virtual void precision(int) {}
  virtual void flush() {}
  virtual size_t tell() {return 0;}
  virtual void seek(size_t) {}
  
  void unsupported(const char *rw, const char *type) {
    std::ostringstream buf;
    buf << rw << " of type " << type << " not supported in " << Mode()
	<< " mode.";
    reportError(buf);
  }
  
  void noread(const char *type) {unsupported("Read",type);}
  void nowrite(const char *type) {unsupported("Write",type);}
  
  virtual void read(bool&) {noread("bool");}
  virtual void read(int&) {noread("int");}
  virtual void read(double&) {noread("real");}
  virtual void read(float&) {noread("real");}
  virtual void read(pair&) {noread("pair");}
  virtual void read(triple&) {noread("triple");}
  virtual void read(char&) {noread("char");}
  virtual void readwhite(mem::string&) {noread("string");}
  virtual void read(mem::string&) {noread("string");}
  
  virtual void write(bool) {nowrite("bool");}
  virtual void write(int) {nowrite("int");}
  virtual void write(double) {nowrite("real");}
  virtual void write(const pair&) {nowrite("pair");}
  virtual void write(const triple&) {nowrite("triple");}
  virtual void write(const mem::string&) {nowrite("string");}
  virtual void write(const pen&) {nowrite("pen");}
  virtual void write(guide *) {nowrite("guide");}
  virtual void write(const transform&) {nowrite("transform");}
  virtual void writeline() {nowrite("string");}
  
  int Nx() {return nx;}
  int Ny() {return ny;}
  int Nz() {return nz;}
  
  void LineMode(bool b) {linemode=b;}
  bool LineMode() {return linemode;}
  
  void CSVMode(bool b) {csvmode=b;}
  bool CSVMode() {return csvmode;}
  
  void SingleMode(bool b) {singlemode=b;}
  bool SingleMode() {return singlemode;}
};

class ifile : public file {
  istream *stream;
  std::ifstream fstream;
  bool first;
  char comment;
  bool comma,nullfield; // Used to detect a final null field in cvs+line mode.
  mem::string whitespace;
  
public:
  ifile(const string& name, bool check=true, char comment=0)
    : file(name,check), comment(comment), comma(false), nullfield(false) {
      stream=&std::cin;
  }
  
  ~ifile() {close();}
  
  void open() {
    if(standard) {
      stream=&std::cin;
    } else {
      fstream.open(name.c_str());
      stream=&fstream;
      if(checkappend) Check();
    }
    first=true;
  }
  
  void seek(size_t pos) {
    if(!standard && !closed) fstream.seekg(pos);
  }
  
  size_t tell() {
    return fstream.tellg();
  }
  
  const char* Mode() {return "input";}
  
  void csv();
  
  void ignoreComment(bool readstring=false);
  bool eol();
  bool nexteol();
  
  bool text() {return true;}
  bool eof() {return stream->eof();}
  bool error() {return stream->fail();}
  void close() {if(!standard && !closed) {fstream.close(); closed=true;}}
  void clear() {stream->clear();}
  
public:

  mem::string getcsvline();
  
  // Skip over white space
  void readwhite(string& val) {val=string(); *stream >> val;}
  
  void Read(bool &val) {string t; readwhite(t); val=(t == "true");}
  void Read(int& val) {*stream >> val;}
  void Read(double& val) {*stream >> val;}
  void Read(pair& val) {*stream >> val;}
  void Read(triple& val) {*stream >> val;}
  void Read(char& val) {stream->get(val);}
  void Read(mem::string& val) {
    if(csvmode) {
      val=whitespace+getcsvline();
    } else {
      mem::string s;
      getline(*stream,s);
      val=whitespace+s;
    }
  }
  
  template<class T>
  void iread(T&);
  
  void read(bool& val) {iread<bool>(val);}
  void read(int& val) {iread<int>(val);}
  void read(double& val) {iread<double>(val);}
  void read(pair& val) {iread<pair>(val);}
  void read(triple& val) {iread<triple>(val);}
  void read(char& val) {iread<char>(val);}
  void read(mem::string& val) {iread<mem::string>(val);}
};
  
class ofile : public file {
  std::ostream *stream;
  std::ofstream fstream;
public:
  ofile(const string& name, bool append=false) : file(name,append) {
      stream=&std::cout;
  }
  
  ~ofile() {close();}
  
  void open() {
    if(standard) {
      stream=&std::cout;
    } else {
      fstream.open(name.c_str(),checkappend ? std::ios::app : std::ios::trunc);
      stream=&fstream;
      Check();
    }
  }
  
  void seek(size_t pos) {
    if(!standard && !closed) fstream.seekp(pos);
  }
  
  const char* Mode() {return "output";}

  bool text() {return true;}
  bool eof() {return stream->eof();}
  bool error() {return stream->fail();}
  void close() {if(!standard && !closed) {fstream.close(); closed=true;}}
  void clear() {stream->clear();}
  void precision(int p) {stream->precision(p);}
  void flush() {stream->flush();}
  
  void write(bool val) {*stream << (val ? "true " : "false ");}
  void write(int val) {*stream << val;}
  void write(double val) {*stream << val;}
  void write(const pair& val) {*stream << val;}
  void write(const triple& val) {*stream << val;}
  void write(const mem::string& val) {*stream << val;}
  void write(const pen& val) {*stream << val;}
  void write(guide *val) {*stream << *val;}
  void write(const transform& val) {*stream << val;}
  void writeline() {
    if(standard && interact::interactive) {
      int scroll=settings::getScroll();
      if(scroll && lines > 0 && lines % scroll == 0) {
	for(;;) {
	  if(!std::cin.good()) {
	    *stream << newline;
	    std::cin.clear();
	    break;
	  }
	  int c=std::cin.get();
	  if(c == '\n') break;
	  while(std::cin.get() != '\n'); // Discard any additional characters
	  if(c == 'q') throw quit();
	}
      } else *stream << newline;
      ++lines;
    } else *stream << newline;
    if(errorstream::interrupt) throw interrupted();
  }
};

#ifdef HAVE_RPC_RPC_H

class ixfile : public file {
  xdr::ixstream stream;
public:
  ixfile(const string& name, bool check=true) : 
    file(name,check), stream(name.c_str()) {if(check) Check();}

  ~ixfile() {close();}
  
  const char* Mode() {return "xinput";}
  
  bool eof() {return stream.eof();}
  bool error() {return stream.fail();}
  void close() {if(!closed) {stream.close(); closed=true;}}
  void clear() {stream.clear();}
  
  void read(int& val) {val=0; stream >> val;}
  void read(double& val) {
    if(singlemode) {float fval=0.0; stream >> fval; val=fval;}
    else {
      val=0.0;
      stream >> val;
    }
  }
  void read(pair& val) {
    double x=0.0, y=0.0;
    stream >> x >> y;
    val=pair(x,y);
  }
  void read(triple& val) {
    double x=0.0, y=0.0, z=0.0;
    stream >> x >> y >> z;
    val=triple(x,y,z);
  }
};

class oxfile : public file {
  xdr::oxstream stream;
public:
  oxfile(const string& name, bool append=false) : 
    file(name), stream(name.c_str(),
		       append ? xdr::xios::app : xdr::xios::trunc) {Check();}

  ~oxfile() {close();}
  
  const char* Mode() {return "xoutput";}
  
  bool eof() {return stream.eof();}
  bool error() {return stream.fail();}
  void close() {if(!closed) {stream.close(); closed=true;}}
  void clear() {stream.clear();}
  void flush() {stream.flush();}
  
  void write(int val) {stream << val;}
  void write(double val) {
    if(singlemode) {float fval=val; stream << fval;}
    stream << val;
  }
  void write(const pair& val) {
    stream << val.getx() << val.gety();
  }
  void write(const triple& val) {
    stream << val.getx() << val.gety() << val.getz();
  }
};

#endif

extern ofile Stdout;
extern ofile nullfile;

template<class T>
void ifile::iread(T& val)
{
  if(standard) clear();
  if(errorstream::interrupt) throw interrupted();
  else {
    ignoreComment(typeid(T)==typeid(mem::string));
    val=T();
    if(!nullfield)
      Read(val);
    csv();
    whitespace="";
  }
}

} // namespace camp

#endif // FILEIO_H
