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

#include "common.h"
#include "pair.h"
#include "triple.h"
#include "guide.h"
#include "pen.h"

#include "camperror.h"
#include "interact.h"
#include "errormsg.h"
#include "util.h"

namespace vm {
  extern bool indebugger;  
}

namespace camp {

extern string tab;
extern string newline;
  
class file : public gc {
protected:  
  string name;
  int nx,ny,nz;    // Array dimensions
  bool linemode;   // Array reads will stop at eol instead of eof.
  bool csvmode;    // Read comma-separated values.
  bool wordmode;   // Delimit strings by white space instead of eol.
  bool singlemode; // Read/write single-precision XDR/binary values.
  bool closed;     // File has been closed.
  bool checkerase; // Check input for errors/erase output.
  bool standard;   // Standard input/output
  bool binary;     // Read in binary mode.
  int lines;       // Number of scrolled lines
  
  bool nullfield;  // Used to detect a final null field in csv+line mode.
  string whitespace;
public: 

  void resetlines() {lines=0;}
  
  bool Standard() {return standard;}
  
  void standardEOF() {
#if defined(HAVE_LIBREADLINE) && defined(HAVE_LIBCURSES)
    cout << endl;
#endif	
  }
  
  template<class T>
  void purgeStandard(T&) {
    if(standard) {
      int c;
      if(cin.eof())
	standardEOF();
      else {
	cin.clear();
	while((c=cin.peek()) != EOF) {
	  cin.ignore();
	  if(c == '\n') break;
	}
      }
    }
  }
  
  void purgeStandard(string&) {
    if(cin.eof())
      standardEOF();
  }
  
  void dimension(int Nx=-1, int Ny=-1, int Nz=-1) {nx=Nx; ny=Ny; nz=Nz;}
  
  file(const string& name, bool checkerase=true, bool binary=false,
       bool closed=false) :
    name(name), linemode(false), csvmode(false), singlemode(false),
    closed(closed), checkerase(checkerase), standard(name.empty()),
    binary(binary), lines(0), nullfield(false), whitespace("") {dimension();}
  
  virtual void open() {}
  
  void Check() {
    if(error()) {
      ostringstream buf;
      buf << "Cannot open file \"" << name << "\".";
      reportError(buf);
    }
  }
  
  virtual ~file() {}

  virtual const char* Mode() {return "";}

  bool isOpen() {
    if(closed) {
      ostringstream buf;
      buf << "I/O operation attempted on ";
      if(name != "") buf << "closed file \'" << name << "\'.";
      else buf << "null file.";
      reportError(buf);
    }
    return true;
  }
		
  string filename() {return name;}
  virtual bool eol() {return false;}
  virtual bool nexteol() {return false;}
  virtual bool text() {return false;}
  virtual bool eof() {return true;}
  virtual bool error() {return true;}
  virtual void close() {}
  virtual void clear() {}
  virtual void precision(int) {}
  virtual void flush() {}
  virtual size_t tell() {return 0;}
  virtual void seek(int, bool=true) {}
  
  void unsupported(const char *rw, const char *type) {
    ostringstream buf;
    buf << rw << " of type " << type << " not supported in " << Mode()
	<< " mode.";
    reportError(buf);
  }
  
  void noread(const char *type) {unsupported("Read",type);}
  void nowrite(const char *type) {unsupported("Write",type);}
  
  virtual void Read(bool&) {noread("bool");}
  virtual void Read(int&) {noread("int");}
  virtual void Read(double&) {noread("real");}
  virtual void Read(float&) {noread("real");}
  virtual void Read(pair&) {noread("pair");}
  virtual void Read(triple&) {noread("triple");}
  virtual void Read(char&) {noread("char");}
  virtual void Read(string&) {noread("string");}
  virtual void readwhite(string&) {noread("string");}
  
  virtual void write(bool) {nowrite("bool");}
  virtual void write(int) {nowrite("int");}
  virtual void write(double) {nowrite("real");}
  virtual void write(const pair&) {nowrite("pair");}
  virtual void write(const triple&) {nowrite("triple");}
  virtual void write(const string&) {nowrite("string");}
  virtual void write(const pen&) {nowrite("pen");}
  virtual void write(guide *) {nowrite("guide");}
  virtual void write(const transform&) {nowrite("transform");}
  virtual void writeline() {nowrite("string");}
  
  virtual void ignoreComment(bool=false) {};
  virtual void csv() {};
  
  template<class T>
  void ignoreComment(T&) {
    ignoreComment();
  }
  
  void ignoreComment(string&) {}
  void ignoreComment(char&) {}
  
  template<class T>
  void read(T& val) {
    if(binary) Read(val);
    else {
      if(standard) clear();
      if(errorstream::interrupt) throw interrupted();
      else {
	ignoreComment(val);
	val=T();
	if(!nullfield)
	  Read(val);
	csv();
	whitespace="";
      }
    }
  }
  
  int Nx() {return nx;}
  int Ny() {return ny;}
  int Nz() {return nz;}
  
  void LineMode(bool b) {linemode=b;}
  bool LineMode() {return linemode;}
  
  void CSVMode(bool b) {csvmode=b; if(b) wordmode=false;}
  bool CSVMode() {return csvmode;}
  
  void WordMode(bool b) {wordmode=b; if(b) csvmode=false;}
  bool WordMode() {return wordmode;}
  
  void SingleMode(bool b) {singlemode=b;}
  bool SingleMode() {return singlemode;}
};

class ifile : public file {
protected:  
  istream *stream;
  std::fstream fstream;
  char comment;
  bool comma;
  
public:
  ifile(const string& name, char comment, bool check=true)
    : file(name,check), comment(comment), comma(false) {stream=&cin;}
  
  // Binary file
  ifile(const string& name, bool check=true) : file(name,check,true) {}
  
  ~ifile() {close();}
  
  void open() {
    if(standard) {
      stream=&cin;
    } else {
      fstream.open(name.c_str());
      stream=&fstream;
      if(checkerase) Check();
    }
  }
  
  const char* Mode() {return "input";}
  
  bool eol();
  bool nexteol();
  
  bool text() {return true;}
  bool eof() {return stream->eof();}
  bool error() {return stream->fail();}
  void close() {if(!standard && !closed) {fstream.close(); closed=true;}}
  void clear() {stream->clear();}
  
  void seek(int pos, bool begin=true) {
    if(!standard && !closed) {
      clear();
      fstream.seekg(pos,begin ? std::ios::beg : std::ios::end);
    }
  }
  
  size_t tell() {return fstream.tellg();}
  
  void csv();
  
  virtual void ignoreComment(bool readstring=false);
  
  // Skip over white space
  void readwhite(string& val) {val=string(); *stream >> val;}
  
  void Read(bool &val) {string t; readwhite(t); val=(t == "true");}
  void Read(int& val) {*stream >> val;}
  void Read(double& val) {*stream >> val;}
  void Read(pair& val) {*stream >> val;}
  void Read(triple& val) {*stream >> val;}
  void Read(char& val) {stream->get(val);}
  void Read(string& val);
};
  
class iofile : public ifile {
public:
  iofile(const string& name, char comment=0) : ifile(name,true,comment) {}

  void precision(int p) {stream->precision(p);}
  void flush() {fstream.flush();}
  
  void write(bool val) {fstream << (val ? "true " : "false ");}
  void write(int val) {fstream << val;}
  void write(double val) {fstream << val;}
  void write(const pair& val) {fstream << val;}
  void write(const triple& val) {fstream << val;}
  void write(const string& val) {fstream << val;}
  void write(const pen& val) {fstream << val;}
  void write(guide *val) {fstream << *val;}
  void write(const transform& val) {fstream << val;}
  
  void writeline() {
    fstream << newline;
    if(errorstream::interrupt) throw interrupted();
  }
};
  
class ofile : public file {
protected:
  std::ostream *stream;
  std::ofstream fstream;
public:
  ofile(const string& name) 
    : file(name) {stream=&cout;}
  
  ~ofile() {close();}
  
  void open() {
    checkLocal(name);
    if(standard) {
      stream=&cout;
    } else {
      fstream.open(name.c_str(),std::ios::trunc);
      stream=&fstream;
      Check();
    }
  }
  
  const char* Mode() {return "output";}

  bool text() {return true;}
  bool eof() {return stream->eof();}
  bool error() {return stream->fail();}
  void close() {if(!standard && !closed) {fstream.close(); closed=true;}}
  void clear() {stream->clear();}
  void precision(int p) {stream->precision(p);}
  void flush() {stream->flush();}
  
  void seek(int pos, bool begin=true) {
    if(!standard && !closed) {
      clear();
      fstream.seekp(pos,begin ? std::ios::beg : std::ios::end);
    }
  }
  
  size_t tell() {return fstream.tellp();}
  
  void write(bool val) {*stream << (val ? "true " : "false ");}
  void write(int val) {*stream << val;}
  void write(double val) {*stream << val;}
  void write(const pair& val) {*stream << val;}
  void write(const triple& val) {*stream << val;}
  void write(const string& val) {*stream << val;}
  void write(const pen& val) {*stream << val;}
  void write(guide *val) {*stream << *val;}
  void write(const transform& val) {*stream << val;}
  
  void writeline();
};

class ibfile : public ifile {
protected:  
  std::fstream fstream;
public:
  ibfile(const string& name, bool check=true) : ifile(name,check) {}

  void open() {
    if(standard) {
      reportError("Cannot open standard input in binary mode");
    } else {
      fstream.open(name.c_str(),std::ios::binary |
		   std::ios::in | std::ios::out);
      stream=&fstream;
      if(checkerase) Check();
    }
  }
  
  template<class T>
  void iread(T& val) {
    val=T();
    fstream.read((char *) &val,sizeof(T));
  }
  
  void Read(bool& val) {iread(val);}
  void Read(int& val) {iread(val);}
  void Read(char& val) {iread(val);}
  void Read(string& val) {iread(val);}
  
  void Read(double& val) {
    if(singlemode) {float fval=0.0; iread(fval); val=fval;}
    else iread(val);
  }
};
  
class iobfile : public ibfile {
public:
  iobfile(const string& name) : ibfile(name,true) {}

  void flush() {fstream.flush();}
  
  template<class T>
  void iwrite(T val) {
    fstream.write((char *) &val,sizeof(T));
  }
  
  void write(bool val) {iwrite(val);}
  void write(int val) {iwrite(val);}
  void write(const string& val) {iwrite(val);}
  void write(const pen& val) {iwrite(val);}
  void write(guide *val) {iwrite(val);}
  void write(const transform& val) {iwrite(val);}
  void write(double val) {
    if(singlemode) {float fval=val; iwrite(fval);}
    else iwrite(val);
  }
  void write(const pair& val) {
    write(val.getx());
    write(val.gety());
  }
  void write(const triple& val) {
    write(val.getx());
    write(val.gety());
    write(val.getz());
  }
  void writeline() {}
};
  
class obfile : public ofile {
public:
  obfile(const string& name) : ofile(name) {}

  void open() {
    checkLocal(name);
    if(standard) {
      reportError("Cannot open standard output in binary mode");
    } else {
      fstream.open(name.c_str(),std::ios::binary | std::ios::trunc);
      stream=&fstream;
      Check();
    }
  }
  
  template<class T>
  void iwrite(T val) {
    fstream.write((char *) &val,sizeof(T));
  }
  
  void write(bool val) {iwrite(val);}
  void write(int val) {iwrite(val);}
  void write(const string& val) {iwrite(val);}
  void write(const pen& val) {iwrite(val);}
  void write(guide *val) {iwrite(val);}
  void write(const transform& val) {iwrite(val);}
  void write(double val) {
    if(singlemode) {float fval=val; iwrite(fval);}
    else iwrite(val);
  }
  void write(const pair& val) {
    write(val.getx());
    write(val.gety());
  }
  void write(const triple& val) {
    write(val.getx());
    write(val.gety());
    write(val.getz());
  }
  
  void writeline() {}
};
  
#ifdef HAVE_RPC_RPC_H

class ixfile : public file {
protected:  
  xdr::ioxstream stream;
public:
  ixfile(const string& name, bool check=true,
	 xdr::xios::open_mode mode=xdr::xios::in) :
    file(name,check,true), stream(name.c_str(), mode) {
    if(check) Check();
  }

  ~ixfile() {close();}
  
  const char* Mode() {return "xinput";}
  
  bool eof() {return stream.eof();}
  bool error() {return stream.fail();}
  void close() {if(!closed) {stream.close(); closed=true;}}
  void clear() {stream.clear();}
  
  void seek(int pos, bool begin=true) {
    if(!standard && !closed) {
      clear();
      stream.seek(pos,begin ? xdr::xios::beg : xdr::xios::end);
    }
  }
  
  size_t tell() {return stream.tell();}
  
  void Read(int& val) {val=0; stream >> val;}
  void Read(double& val) {
    if(singlemode) {float fval=0.0; stream >> fval; val=fval;}
    else {
      val=0.0;
      stream >> val;
    }
  }
  void Read(pair& val) {
    double x,y;
    Read(x);
    Read(y);
    val=pair(x,y);
  }
  void Read(triple& val) {
    double x,y,z;
    Read(x);
    Read(y);
    Read(z);
    val=triple(x,y,z);
  }
};

class ioxfile : public ixfile {
public:
  ioxfile(const string& name) : ixfile(name,true,xdr::xios::out) {}

  void flush() {stream.flush();}
  
  void write(int val) {stream << val;}
  void write(double val) {
    if(singlemode) {float fval=val; stream << fval;}
    else stream << val;
  }
  void write(const pair& val) {
    write(val.getx());
    write(val.gety());
  }
  void write(const triple& val) {
    write(val.getx());
    write(val.gety());
    write(val.getz());
  }
};
  
class oxfile : public file {
  xdr::oxstream stream;
public:
  oxfile(const string& name) : 
    file(name), stream((checkLocal(name),name.c_str()),xdr::xios::trunc) {
    Check();
  }

  ~oxfile() {close();}
  
  const char* Mode() {return "xoutput";}
  
  bool eof() {return stream.eof();}
  bool error() {return stream.fail();}
  void close() {if(!closed) {stream.close(); closed=true;}}
  void clear() {stream.clear();}
  void flush() {stream.flush();}
  
  void seek(int pos, bool begin=true) {
    if(!standard && !closed) {
      clear();
      stream.seek(pos,begin ? xdr::xios::beg : xdr::xios::end);
    }
  }
  
  size_t tell() {return stream.tell();}
  
  void write(int val) {stream << val;}
  void write(double val) {
    if(singlemode) {float fval=val; stream << fval;}
    else stream << val;
  }
  void write(const pair& val) {
    write(val.getx());
    write(val.gety());
  }
  void write(const triple& val) {
    write(val.getx());
    write(val.gety());
    write(val.getz());
  }
};

#endif

extern ofile Stdout;
extern file nullfile;

} // namespace camp

#endif // FILEIO_H
