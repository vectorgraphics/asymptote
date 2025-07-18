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
#include <zlib.h>

#include "common.h"

#ifdef HAVE_LIBTIRPC
#include "xstream.h"
#endif

#include "pair.h"
#include "triple.h"
#include "guide.h"
#include "pen.h"

#include "camperror.h"
#include "interact.h"
#include "errormsg.h"
#include "util.h"
#include "asyprocess.h"
#include "locate.h"
#include "asyparser.h"

namespace vm {
extern bool indebugger;
}

namespace camp {

extern string tab;
extern string newline;

enum Mode {NOMODE,INPUT,OUTPUT,UPDATE,BINPUT,BOUTPUT,BUPDATE,
  XINPUT,XOUTPUT,XUPDATE,XINPUTGZ,XOUTPUTGZ,OPIPE};

static const string FileModes[]=
{"none","input","output","output(update)",
 "input(binary)","output(binary)","output(binary,update)",
 "input(xdr)","output(xdr)","output(xdr,update)",
 "input(xdrgz)","output(xdrgz)","output(pipe)"};

extern FILE *pipeout;

void openpipeout();
string locatefile(string name);

class file : public gc {
protected:
  string name;
  bool check;      // Check whether input file exists.
  Mode type;

  Int nx,ny,nz;    // Array dimensions
  bool linemode;   // Array reads will stop at eol instead of eof.
  bool csvmode;    // Read comma-separated values.
  bool wordmode;   // Delimit strings by white space instead of eol.
  bool singlereal; // Read/write single-precision XDR/binary reals.
  bool singleint;  // Read/write single-precision XDR/binary ints.
  bool signedint;  // Read/write signed XDR/binary ints.

  bool closed;     // File has been closed.
  bool standard;   // Standard input/output
  bool binary;     // Read in binary mode.

  bool nullfield;  // Used to detect a null field in cvs mode.
  string whitespace;
  size_t index;    // Terminator index.

public:

  bool Standard();
  bool enabled() {return !standard || settings::verbose > 1 ||
      interact::interactive || !settings::getSetting<bool>("quiet");}

  void standardEOF();

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

  void purgeStandard(string&);

  void dimension(Int Nx=-1, Int Ny=-1, Int Nz=-1);

  file(const string& name, bool check=true, Mode type=NOMODE,
       bool binary=false, bool closed=false);

  virtual bool isBinary() {return false;}

  virtual bool isXDR() {return false;}

  virtual void open() {}

  void Check();

  virtual ~file();

  bool isOpen();

  string filename() {return name;}
  virtual bool eol() {return false;}
  virtual bool nexteol() {return false;}
  virtual bool text() {return false;}
  virtual bool eof() {return true;}
  virtual bool error() {return true;}
  virtual void close() {}
  virtual void clear() {}
  virtual Int precision(Int) {return 0;}
  virtual void flush() {}
  virtual size_t tell() {return 0;}
  virtual void seek(Int, bool=true) {}

  string FileMode() {return FileModes[type];}

  void unsupported(const char *rw, const char *type);

  void noread(const char *type) {unsupported("Read",type);}
  void nowrite(const char *type) {unsupported("Write",type);}

  virtual void Read(bool&) {noread("bool");}
  virtual void Read(Int&) {noread("int");}
  virtual void Read(double&) {noread("real");}
  virtual void Read(float&) {noread("real");}
  virtual void Read(pair&) {noread("pair");}
  virtual void Read(triple&) {noread("triple");}
  virtual void Read(char&) {noread("char");}
  virtual void Read(string&) {noread("string");}
  virtual void readwhite(string&) {noread("string");}

  virtual void write(bool) {nowrite("bool");}
  virtual void write(char) {nowrite("char");}
  virtual void write(Int) {nowrite("int");}
  virtual void write(double) {nowrite("real");}
  virtual void write(const pair&) {nowrite("pair");}
  virtual void write(const triple&) {nowrite("triple");}
  virtual void write(const string&) {nowrite("string");}
  virtual void write(const pen&) {nowrite("pen");}
  virtual void write(guide *) {nowrite("guide");}
  virtual void write(const transform&) {nowrite("transform");}
  virtual void writeline() {nowrite("string");}

  virtual void ignoreComment() {};
  virtual void csv() {};

  template<class T>
  void ignoreComment(T&) {
    ignoreComment();
  }

  void ignoreComment(string&) {}
  void ignoreComment(char&) {}

  template<class T>
  void setDefault(T& val) {
    val=T();
  }

#if COMPACT
  void setDefault(Int& val) {
    val=vm::Undefined;
  }
#endif

  template<class T>
  void read(T& val) {
    if(binary) Read(val);
    else {
      if(standard) clear();
      if(errorstream::interrupt) throw interrupted();
      else {
        ignoreComment(val);
        setDefault(val);
        if(!nullfield)
          Read(val);
        csv();
        whitespace="";
      }
    }
  }

  Int Nx() {return nx;}
  Int Ny() {return ny;}
  Int Nz() {return nz;}

  void Nx(Int n) {nx=n;}
  void Ny(Int n) {ny=n;}
  void Nz(Int n) {nz=n;}

  void LineMode(bool b) {linemode=b;}
  bool LineMode() {return linemode;}

  void CSVMode(bool b) {csvmode=b; if(b) wordmode=false;}
  bool CSVMode() {return csvmode;}

  void WordMode(bool b) {wordmode=b; if(b) csvmode=false;}
  bool WordMode() {return wordmode;}

  void SingleReal(bool b) {singlereal=b;}
  bool SingleReal() {return singlereal;}

  void SingleInt(bool b) {singleint=b;}
  bool SingleInt() {return singleint;}

  void SignedInt(bool b) {signedint=b;}
  bool SignedInt() {return signedint;}
};

class opipe : public file {
public:
  opipe(const string& name) : file(name,false,OPIPE) {standard=false;}

  void open() {
    openpipeout();
  }

  bool text() {return true;}
  bool eof() {return pipeout ? feof(pipeout) : true;}
  bool error() {return pipeout ? ferror(pipeout) : true;}
  void clear() {if(pipeout) clearerr(pipeout);}
  void flush();

  void seek(Int pos, bool begin=true) {
    if(!standard && pipeout) {
      clear();
      fseek(pipeout,pos,begin ? SEEK_SET : SEEK_END);
    }
  }

  size_t tell() {
    return pipeout ? ftell(pipeout) : 0;
  }

  void write(const string& val);

  void write(bool val) {
    ostringstream s;
    s << val;
    write(s.str());
  }

  void write(Int val) {
    ostringstream s;
    s << val;
    write(s.str());
  }
  void write(double val) {
    ostringstream s;
    s << val;
    write(s.str());
  }
  void write(const pair& val) {
    ostringstream s;
    s << val;
    write(s.str());
  }
  void write(const triple& val) {
    ostringstream s;
    s << val;
    write(s.str());
  }

  void write(const pen &val) {
    ostringstream s;
    s << val;
    write(s.str());
  }

  void write(guide *val) {
    ostringstream s;
    s << *val;
    write(s.str());
  }

  void write(const transform& val) {
    ostringstream s;
    s << val;
    write(s.str());
  }

  void writeline() {
    fprintf(pipeout,"\n");
    if(errorstream::interrupt) throw interrupted();
  }
};

class ifile : public file {
protected:
  istream *stream;
  std::fstream *fstream;
  stringstream buf;
  char comment;
  std::ios::openmode mode;
  bool comma;

public:
  ifile(const string& name, char comment, bool check=true, Mode type=INPUT,
        std::ios::openmode mode=std::ios::in) :
    file(name,check,type), stream(&cin), fstream(NULL),
    comment(comment), mode(mode), comma(false) {}

  // Binary file
  ifile(const string& name, bool check=true, Mode type=BINPUT,
        std::ios::openmode mode=std::ios::in) :
    file(name,check,type,true), mode(mode) {}

  ~ifile() {close();}

  void open();
  bool eol();
  bool nexteol();

  bool text() {return true;}
  bool eof() {return stream->eof();}
  bool error() {return stream->fail();}

  void close() {
    if(!standard && fstream) {
      fstream->close();
      closed=true;
      delete fstream;
      fstream=NULL;
      processData().ifile.remove(index);
    }
  }

  void clear() {stream->clear();}

  void seek(Int pos, bool begin=true) {
    if(!standard && fstream) {
      clear();
      fstream->seekg(pos,begin ? std::ios::beg : std::ios::end);
    }
  }

  size_t tell() {
    if(fstream)
      return fstream->tellg();
    else
      return 0;
  }

  void csv();

  virtual void ignoreComment();

  // Skip over white space
  void readwhite(string& val) {val=string(); *stream >> val;}

  void Read(bool &val) {string t; readwhite(t); val=(t == "true");}
  void Read(Int& val) {*stream >> val;}
  void Read(double& val);
  void Read(pair& val) {*stream >> val;}
  void Read(triple& val) {*stream >> val;}
  void Read(char& val) {stream->get(val);}
  void Read(string& val);
};

class iofile : public ifile {
public:
  iofile(const string& name, char comment=0) :
    ifile(name,comment,true,UPDATE,std::ios::in | std::ios::out) {}

  Int precision(Int p) {
    return p == 0 ? stream->precision(settings::getSetting<Int>("digits")) :
      stream->precision(p);
  }
  void flush() {if(fstream) fstream->flush();}

  void write(bool val) {*fstream << (val ? "true " : "false ");}
  void write(Int val) {*fstream << val;}
  void write(double val) {*fstream << val;}
  void write(const pair& val) {*fstream << val;}
  void write(const triple& val) {*fstream << val;}
  void write(const string& val) {*fstream << val;}
  void write(const pen& val) {*fstream << val;}
  void write(guide *val) {*fstream << *val;}
  void write(const transform& val) {*fstream << val;}

  void writeline();
};

class ofile : public file {
protected:
  ostream *stream;
  std::ofstream *fstream;
  std::ios::openmode mode;
public:
  ofile(const string& name, Mode type=OUTPUT,
        std::ios::openmode mode=std::ios::trunc) :
    file(name,true,type), stream(&cout), fstream(NULL), mode(mode) {}

  ~ofile() {close();}

  void open();

  bool text() {return true;}
  bool eof() {return stream->eof();}
  bool error() {return stream->fail();}

  void close();
  void clear() {stream->clear();}
  Int precision(Int p);
  void flush() {stream->flush();}

  void seek(Int pos, bool begin=true);

  size_t tell();

  bool enabled();

  void write(bool val) {*stream << (val ? "true " : "false ");}
  void write(Int val) {*stream << val;}
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
public:
  ibfile(const string& name, bool check=true, Mode type=BINPUT,
         std::ios::openmode mode=std::ios::in) :
    ifile(name,check,type,mode | std::ios::binary) {}

  bool isBinary() {return true;}

  template<class T>
  void iread(T& val) {
    val=T();
    if(fstream) fstream->read((char *) &val,sizeof(T));
  }

  void Read(bool& val) {iread(val);}
  void Read(Int& val) {
    if(signedint) {
      if(singleint) {int ival; iread(ival); val=ival;}
      else iread(val);
    } else {
      if(singleint) {unsigned ival; iread(ival); val=Intcast(ival);}
      else {unsignedInt ival; iread(ival); val=Intcast(ival);}
    }
  }
  void Read(char& val) {iread(val);}
  void Read(string& val) {
    size_t n=0;
    if(wordmode)
      iread(n);
    else
      n=SIZE_MAX;
    val="";
    string s;
    for(size_t i=0; i < n; ++i) {
      char c;
      Read(c);
      if(eof() || error())
        return;
      s += c;
    }
    val=s;
  }

  void Read(double& val) {
    if(singlereal) {float fval; iread(fval); val=fval;}
    else iread(val);
  }
};

class iobfile : public ibfile {
public:
  iobfile(const string& name) :
    ibfile(name,true,BUPDATE,std::ios::in | std::ios::out) {}

  bool isBinary() {return true;}

  void flush() {if(fstream) fstream->flush();}

  template<class T>
  void iwrite(T val) {
    if(fstream) fstream->write((char *) &val,sizeof(T));
  }

  void write(bool val) {iwrite(val);}
  void write(Int val) {
    if(signedint) {
      if(singleint) iwrite(intcast(val));
      else iwrite(val);
    } else {
      if(singleint) iwrite(unsignedcast(val));
      else iwrite(unsignedIntcast(val));
    }
  }
  void write(const string& val) {
    size_t n=val.size();
    if(wordmode)
      iwrite(n);
    for(size_t i=0; i < n; ++i)
      fstream->write((char *) &val[i],1);
  }
  void write(const pen& val) {iwrite(val);}
  void write(guide *val) {iwrite(val);}
  void write(const transform& val) {iwrite(val);}
  void write(double val) {
    if(singlereal) iwrite((float) val);
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
  obfile(const string& name) : ofile(name,BOUTPUT,std::ios::binary) {}

  bool isBinary() {return true;}

  template<class T>
  void iwrite(T val) {
    if(fstream) fstream->write((char *) &val,sizeof(T));
  }

  void write(bool val) {iwrite(val);}
  void write(Int val) {
    if(signedint) {
      if(singleint) iwrite(intcast(val));
      else iwrite(val);
    } else {
      if(singleint) iwrite(unsignedcast(val));
      else iwrite(unsignedIntcast(val));
    }
  }
  void write(const string& val) {
    size_t n=val.size();
    if(wordmode)
      iwrite(n);
    for(size_t i=0; i < n; ++i)
      fstream->write((char *) &val[i],1);
  }
  void write(const pen& val) {iwrite(val);}
  void write(guide *val) {iwrite(val);}
  void write(const transform& val) {iwrite(val);}
  void write(double val) {
    if(singlereal) iwrite((float) val);
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

#ifdef HAVE_LIBTIRPC

class ixfile : public file {
protected:
  xdr::ixstream *fstream;
  xdr::xios::open_mode mode;
public:
  ixfile(const string& name, bool check=true, Mode type=XINPUT,
         xdr::xios::open_mode mode=xdr::xios::in) :
    file(name,check,type,true), fstream(NULL), mode(mode) {}

  bool isXDR() override {return true;}

  void open() override {
    name=locatefile(inpath(name));
    fstream=new xdr::ixstream(name.c_str(),mode);
    index=processData().ixfile.add(fstream);
    if(check) Check();
  }

  void close() override {
    if(fstream) {
      fstream->close();
      closed=true;
      delete fstream;
      fstream=NULL;
      processData().ixfile.remove(index);
    }
  }

  ~ixfile() {close();}

  bool eof() override {return fstream ? fstream->eof() : true;}
  bool error() override {return fstream ? fstream->fail() : true;}

  void clear() override {if(fstream) fstream->clear();}

  void seek(Int pos, bool begin=true) override {
    if(!standard && fstream) {
      clear();
      fstream->seek(pos,begin ? xdr::xios::beg : xdr::xios::end);
    }
  }

  size_t tell() override {
    if(fstream)
      return fstream->tell();
    else
      return 0;
  }

  void Read(char& val) override {
    xdr::xbyte b;
    *fstream >> b;
    val=b;
  }

  void Read(string& val) override {
    size_t n=0;
    if(wordmode)
      *fstream >> n;
    else
      n=SIZE_MAX;
    val="";
    string s;
    for(size_t i=0; i < n; ++i) {
      char c;
      Read(c);
      if(eof() || error())
        return;
      s += c;
    }
    val=s;
  }

  void Read(Int& val) override {
    if(signedint) {
      if(singleint) {int ival=0; *fstream >> ival; val=ival;}
      else {val=0; *fstream >> val;}
    } else {
      if(singleint) {unsigned ival=0; *fstream >> ival; val=Intcast(ival);}
      else {unsignedInt ival=0; *fstream >> ival; val=Intcast(ival);}
    }
  }
  void Read(double& val) override {
    if(singlereal) {float fval=0.0; *fstream >> fval; val=fval;}
    else {
      val=0.0;
      *fstream >> val;
    }
  }
  void Read(pair& val) override {
    double x,y;
    Read(x);
    Read(y);
    val=pair(x,y);
  }
  void Read(triple& val) override {
    double x,y,z;
    Read(x);
    Read(y);
    Read(z);
    val=triple(x,y,z);
  }
};

class igzxfile : public ixfile {
protected:
  std::vector<uint8_t> readData;
  size_t const readSize;
  gzFile gzfile;
public:
  igzxfile(const string& name, bool check=true,
           xdr::xios::open_mode mode=xdr::xios::in, size_t readSize=32768) :
    ixfile(name,check,XINPUTGZ,mode), readSize(readSize) {}

  bool error() override {return !gzfile;}

  void open() override;

  void close() override {
    closeFile();
  }

  ~igzxfile() override {closeFile();}


protected:
  void closeFile();
};

class ioxfile : public ixfile {
public:
  ioxfile(const string& name) :
    ixfile(outpath(name),true,XUPDATE,xdr::xios::out) {}

   void open() override {
    name=locatefile(inpath(name));
    ioxfstreamRef=new xdr::ioxstream(name.c_str(),mode);
    fstream=static_cast<xdr::ixstream*>(ioxfstreamRef);
    index=processData().ixfile.add(fstream);
    if(check) Check();
  }

  void flush() override {if(fstream) ioxfstreamRef->flush();}

  void write(const string& val) override {
    size_t n=val.size();
    if(wordmode)
      *ioxfstreamRef << n;
    for(size_t i=0; i < n; ++i)
      *ioxfstreamRef << (xdr::xbyte) val[i];
  }

  void write(Int val) override {
    if(signedint) {
      if(singleint) *ioxfstreamRef << intcast(val);
      else *ioxfstreamRef << val;
    } else {
      if(singleint) *ioxfstreamRef << unsignedcast(val);
      else *ioxfstreamRef << unsignedIntcast(val);
    }
  }
  void write(double val) override {
    if(singlereal) *ioxfstreamRef << (float) val;
    else *ioxfstreamRef << val;
  }
  void write(const pair& val) override {
    write(val.getx());
    write(val.gety());
  }
  void write(const triple& val) override {
    write(val.getx());
    write(val.gety());
    write(val.getz());
  }

private:
  xdr::ioxstream* ioxfstreamRef;
};

class oxfile : public file {
protected:
  xdr::oxstream *fstream;
public:
  oxfile(const string& name, Mode type=XOUTPUT) : file(name,true,type),
                                                  fstream(NULL) {}

  bool isXDR() override {return true;}

  void open() override {
    fstream=new xdr::oxstream(outpath(name).c_str(),xdr::xios::trunc);
    index=processData().oxfile.add(fstream);
    Check();
  }

  void close() override {
    if(fstream) {
      fstream->close();
      closed=true;
      delete fstream;
      fstream=NULL;
      processData().oxfile.remove(index);
    }
  }

  ~oxfile() {close();}

  bool eof() override {return fstream ? fstream->eof() : true;}
  bool error() override {return fstream ? fstream->fail() : true;}
  void clear() override {if(fstream) fstream->clear();}
  void flush() override {if(fstream) fstream->flush();}

  void seek(Int pos, bool begin=true) override {
    if(!standard && fstream) {
      clear();
      fstream->seek(pos,begin ? xdr::xios::beg : xdr::xios::end);
    }
  }

  size_t tell() override {
    if(fstream)
      return fstream->tell();
    else
      return 0;
  }

  void write(const string& val) override {
    size_t n=val.size();
    if(wordmode)
      *fstream << n;
    for(size_t i=0; i < n; ++i)
      *fstream << (xdr::xbyte) val[i];
  }

  void write(Int val) override {
    if(signedint) {
      if(singleint) *fstream << intcast(val);
      else *fstream << val;
    } else {
      if(singleint) *fstream << unsignedcast(val);
      else *fstream << unsignedIntcast(val);
    }
  }
  void write(double val) override {
    if(singlereal) *fstream << (float) val;
    else *fstream << val;
  }
  void write(const pair& val) override {
    write(val.getx());
    write(val.gety());
  }
  void write(const triple& val) override {
    write(val.getx());
    write(val.gety());
    write(val.getz());
  }
};

class ogzxfile : public oxfile {
  string name;
  bool destroyed;
public:
  xdr::memoxstream memxdrfile;

  ogzxfile(const string& name, bool singleprecision=false) :
    oxfile(name,XOUTPUTGZ), name(name), destroyed(false),
    memxdrfile(singleprecision) {}

  void open() override {
    fstream=&memxdrfile;
  }

  void close() override;

  ~ogzxfile() override {close();}
};

#endif

extern ofile Stdout;
extern file nullfile;

} // namespace camp

#endif // FILEIO_H
