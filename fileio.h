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
#include "guide.h"
#include "pen.h"
#include "camperror.h"

namespace camp {

extern std::string tab;
extern std::string newline;
  
class file {
protected:  
  std::string name;
  int nx,ny,nz;  // Array dimensions
  bool linemode; // If true, array reads will stop at eol instead of eof.
  bool csvmode;  // If true, read comma-separated values.
  bool closed;
public: 

  bool standard() {return name == "";}
  
  void dimension(int Nx=-1, int Ny=-1, int Nz=-1) {nx=Nx; ny=Ny; nz=Nz;}
  
  file(std::string name) : name(name), linemode(false), csvmode(false),
			   closed(false) {dimension();}
  
  void check() {
    if(error()) {
      std::ostringstream buf;
      buf << "Cannot open file \"" << name << "\".";
      reportError(buf.str().c_str());
      closed=true;
    }
  }
  
  virtual ~file() {}

  virtual const char* Mode()=0;

  bool open() {
    if(closed) {
      std::ostringstream buf;
      buf << "I/O operation attempted on closed file \'" << name << "\'.";
      reportError(buf.str().c_str());
      return false;
    } else return true;
  }
		
  std::string filename() {return name;}
  virtual bool eol() {return false;}
  virtual bool text() {return false;}
  virtual bool eof()=0;
  virtual bool error()=0;
  virtual void close()=0;
  virtual void clear()=0;
  virtual void precision(int) {}
  virtual void flush() {}
  
  void unsupported(const char *rw, const char *type) {
    std::ostringstream buf;
    buf << rw << " of type " << type << " not supported in " << Mode()
	<< " mode.";
    reportError(buf.str().c_str());
  }
  
  void noread(const char *type) {unsupported("Read",type);}
  void nowrite(const char *type) {unsupported("Write",type);}
  
  virtual void read(bool&) {noread("bool");}
  virtual void read(int&) {noread("int");}
  virtual void read(double&) {noread("real");}
  virtual void read(pair&) {noread("pair");}
  virtual void read(char&) {noread("char");}
  virtual void readwhite(std::string&) {noread("string");}
  virtual void read(std::string&) {noread("string");}
  
  virtual void write(bool) {nowrite("bool");}
  virtual void write(int) {nowrite("int");}
  virtual void write(double) {nowrite("real");}
  virtual void write(const pair&) {nowrite("pair");}
  virtual void write(const std::string&) {nowrite("string");}
  virtual void write(const pen&) {nowrite("pen");}
  virtual void write(const guide&) {nowrite("guide");}
  virtual void write(const transform&) {nowrite("transform");}
  
  int Nx() {return nx;}
  int Ny() {return ny;}
  int Nz() {return nz;}
  
  void LineMode(bool b) {linemode=b;}
  bool LineMode() {return linemode;}
  
  void CSVMode(bool b) {csvmode=b;}
  bool CSVMode() {return csvmode;}
};

class ifile : public file {
  istream *stream;
  std::ifstream *fstream;
  
public:
  ifile(std::string name) : file(name) {
    if(standard()) {
      stream=&std::cin;
    } else {
      stream=fstream=new std::ifstream;
      fstream->open(name.c_str());
      check();
    }
  }
  
  virtual ~ifile() {
    if(!standard()) delete fstream;
  }
  
  void seek(size_t pos) {
    if(!standard() && !closed) fstream->seekg(pos);
  }
  
  const char* Mode() {return "input";}
  
  void csv();
  
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
  void close() {if(!standard() && !closed) {fstream->close(); closed=true;}}
  void clear() {stream->clear();}
  
public:

  std::string getcsvline();
  
  // Skip over white space
  void readwhite(std::string& val) {val=std::string(); *stream >> val; csv();}
  
  void Read(bool &val) {std::string t; readwhite(t); val=(t == "true"); csv();}
  void Read(int& val) {val=0; *stream >> val; csv();}
  void Read(double& val) {val=0.0; *stream >> val; csv();}
  void Read(pair& val) {val=0.0; *stream >> val; csv();}
  void Read(char& val) {val=char(); stream->get(val); csv();}
  void Read(std::string& val) {
    val=std::string();
    if(csvmode) {
      val=getcsvline();
      csv();
    } else getline(*stream,val);
  }
  
  template<class T>
  void iread(T&);
  
  void read(bool& val) {iread<bool>(val);}
  void read(int& val) {iread<int>(val);}
  void read(double& val) {iread<double>(val);}
  void read(pair& val) {iread<pair>(val);}
  void read(char& val) {iread<char>(val);}
  void read(std::string& val) {iread<std::string>(val);}
};
  
class ofile : public file {
  std::ostream *stream;
  std::ofstream *fstream;
public:
  ofile(std::string name) : file(name) {
    if(standard()) {
      stream=&std::cout;
    } else {
      stream=fstream=new std::ofstream;
      fstream->open(name.c_str());
      check();
    }
  }
  
  virtual ~ofile() {
    if(!standard()) delete fstream;
  }
  
  void seek(size_t pos) {
    if(!standard() && !closed) fstream->seekp(pos);
  }
  
  const char* Mode() {return "output";}

  bool text() {return true;}
  bool eof() {return stream->eof();}
  bool error() {return stream->fail();}
  void close() {if(!standard() && !closed) {fstream->close(); closed=true;}}
  void clear() {stream->clear();}
  void precision(int p) {stream->precision(p);}
  void flush() {stream->flush();}
  
  void write(bool val) {*stream << (val ? "true " : "false ");}
  void write(int val) {*stream << val;}
  void write(double val) {*stream << val;}
  void write(const pair& val) {*stream << val;}
  void write(const std::string& val) {*stream << val;}
  void write(const pen& val) {*stream << val;}
  void write(const guide& val) {*stream << val;}
  void write(const transform& val) {*stream << val;}
};

#ifdef HAVE_RPC_RPC_H

class ixfile : public file {
  xdr::ixstream stream;
public:
  ixfile(std::string name) : file(name), stream(name.c_str()) {check();}  

  const char* Mode() {return "xinput";}
  
  bool eof() {return stream.eof();}
  bool error() {return stream.fail();}
  void close() {if(!closed) {stream.close(); closed=true;}}
  void clear() {stream.clear();}
  
  void read(int& val) {val=0; stream >> val;}
  void read(double& val) {val=0.0; stream >> val;}
  void read(pair& val) {double x=0.0, y=0.0; stream >> x >> y; val=pair(x,y);}
};

class oxfile : public file {
  xdr::oxstream stream;
public:
  oxfile(std::string name) : file(name), stream(name.c_str()) {check();}  

  const char* Mode() {return "xoutput";}
  
  bool eof() {return stream.eof();}
  bool error() {return stream.fail();}
  void close() {if(!closed) {stream.close(); closed=true;}}
  void clear() {stream.clear();}
  void flush() {stream.flush();}
  
  void write(int val) {stream << val;}
  void write(double val) {stream << val;}
  void write(const pair& val) {stream << val.getx() << val.gety();}
};

#endif

extern ofile Stdout;

extern ifile *typein;
extern ofile *typeout;

template<class T>
void ifile::iread(T& val)
{
  if(settings::suppressOutput && standard() && typein) typein->Read(val);
  else {
    Read(val);
    if(standard() && typeout) {
      typeout->write(val);
      typeout->write((std::string)"\n");
      typeout->flush();
    }
  }
}

} // namespace camp

#endif // FILEIO_H
