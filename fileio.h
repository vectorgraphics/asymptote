/******
 * fileio.h
 * Tom Prince 2004/05/10
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
#include "pool.h"
#include "camperror.h"

namespace camp {

extern std::string tab;
extern std::string newline;
  
class file : public mempool::pooled<file> {
protected:  
  std::string name;
  int nx,ny,nz; // Array dimensions
  bool linemode; // If true, array reads will stop at eol instead of eof.
  bool csvmode; // If true, read comma-separated values.
  bool closed;
public: 
  enum mode {
    in, out,
    xin, xout
  };

  static file* open(std::string filename, mode);

  void dimension(int Nx=-1, int Ny=-1, int Nz=-1) {
    nx=Nx; ny=Ny; nz=Nz;
  }
  
  file(std::string name) : name(name), linemode(false), csvmode(false),
		      closed(false) {dimension();}
  
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
		
  
  std::string filename();
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

class ofile : public file
{
  std::ostream *stream;
  std::ofstream fstream;
public:
  ofile(std::string name) : file(name) {
    if(name == "") {
      stream=&std::cout;
      fstream.setstate(std::ios_base::failbit);
    } else {
      fstream.open(name.c_str());
      stream=&fstream;
    }
  }
  
  const char* Mode() {return "output";}

  bool text() {return true;}
  bool eof() {return stream->eof();}
  bool error() {return stream->fail();}
  void close() {if(fstream) {fstream.close(); closed=true;}}
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

extern ofile Stdout;

}

#endif // FILEIO_H
