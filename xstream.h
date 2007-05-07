/* C++ interface to the XDR External Data Representation I/O routines
   Version 1.45
   Copyright (C) 1999-2007 John C. Bowman

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA. */

#ifndef __xstream_h__
#define __xstream_h__ 1

#ifndef _ALL_SOURCE
#define _ALL_SOURCE 1
#endif

#include <cstdio>

#ifdef _POSIX_SOURCE
#undef _POSIX_SOURCE
#include <rpc/rpc.h>
#define _POSIX_SOURCE
#else
#include <rpc/rpc.h>
#endif

namespace xdr {
  
class xbyte {
  unsigned char c;
 public:
  xbyte() {}
  xbyte(unsigned char c0) : c(c0) {}
  xbyte(int c0) : c((unsigned char) c0) {}
  xbyte(unsigned int c0) : c((unsigned char) c0) {}
  int byte() const {return c;}
  operator unsigned char () const {return c;}
};

class xios {
 public:
  enum io_state {goodbit=0, eofbit=1, failbit=2, badbit=4};
  enum open_mode {in=1, out=2, app=8, trunc=16};
  enum seekdir {beg=SEEK_SET, cur=SEEK_CUR, end=SEEK_END};
 private:	
  int _state;
 public:	
  int good() const { return _state == 0; }
  int eof() const { return _state & eofbit; }
  int fail() const { return !good();}
  int bad() const { return _state & badbit; }
  void clear(int state = 0) {_state=state;}
  void set(int flag) {_state |= flag;}
  operator void*() const { return fail() ? (void*)0 : (void*)(-1); }
  int operator!() const { return fail(); }
};

class xstream : public xios {
 protected:
  FILE *buf;
  XDR xdrs;
 public:
  virtual ~xstream() {}
  xstream() {buf=NULL;}
  void xopen(const char *filename, const char *mode, xdr_op xop) {
    clear();
    buf=fopen(filename,mode);
    if(buf) xdrstdio_create(&xdrs, buf, xop);
    else set(badbit);
  }
  void close() {
    if(buf) {
#if !defined(_CRAY) && (!defined(__i386__) || defined(__ELF__))
      xdr_destroy(&xdrs);
#endif			
      fclose(buf);
      buf=NULL;
    }
  }
  void precision(int) {}
  
  xstream& seek(long pos, seekdir dir=beg) {
    if(fseek(buf,pos,dir) != 0) set(failbit); 
    return *this;
  }
  long tell() {
    return ftell(buf);
  }
};

#define IXSTREAM(T,N) ixstream& operator >> (T& x) \
{if(!xdr_##N(&xdrs, &x)) set(eofbit); return *this;}

#if __linux__ && !__ELF__
// Due to a i386-linuxaout bug, cannot generate xdr output for a.out systems.
#define OXSTREAM(T,N) oxstream& operator << (T) {return *this;}
#else
#define OXSTREAM(T,N) oxstream& operator << (T x) \
{ \
  Encode(); \
  if(!xdr_##N(&xdrs, &x)) set(badbit); return *this; \
}
#endif

#define IOXSTREAM(T,N) ioxstream& operator >> (T& x) \
{ \
  Decode(); \
  if(!xdr_##N(&xdrs, &x)) set(eofbit); \
  return *this; \
}
  
class ixstream : virtual public xstream {
 public:
  void open(const char *filename, open_mode=in) {
    xopen(filename,"r",XDR_DECODE);
  }
	
  ixstream() {}
  ixstream(const char *filename) {open(filename);}
  ixstream(const char *filename, open_mode mode) {open(filename,mode);}
  virtual ~ixstream() {close();}
	
  typedef ixstream& (*imanip)(ixstream&);
  ixstream& operator >> (imanip func) { return (*func)(*this); }
	
  IXSTREAM(int,int);
  IXSTREAM(unsigned int,u_int);
  IXSTREAM(long,long);
  IXSTREAM(unsigned long,u_long);
  IXSTREAM(short,short);
  IXSTREAM(unsigned short,u_short);
  IXSTREAM(char,char);
#ifndef _CRAY		
  IXSTREAM(unsigned char,u_char);
#endif		
  IXSTREAM(float,float);
  IXSTREAM(double,double);
	
  ixstream& operator >> (xbyte& x) {
    x=fgetc(buf);
    if(x.byte() == EOF) set(eofbit);
    return *this;
  }
};

class oxstream : public xstream {
protected:  
  bool decode;
 public:
  void open(const char *filename, open_mode mode=trunc) {
    xopen(filename,(mode & app) ? "a" : "w",XDR_ENCODE);
    decode=false;
  }
	
  void Encode() {
    if(decode) {
      xdrstdio_create(&xdrs, buf, XDR_ENCODE);
      decode=false;
    }
  }
  
  oxstream() {}
  oxstream(const char *filename) {open(filename);}
  oxstream(const char *filename, open_mode mode) {open(filename,mode);}
  virtual ~oxstream() {close();}

  oxstream& flush() {if(buf) fflush(buf); return *this;}
	
  typedef oxstream& (*omanip)(oxstream&);
  oxstream& operator << (omanip func) { return (*func)(*this); }
	
  OXSTREAM(int,int);
  OXSTREAM(unsigned int,u_int);
  OXSTREAM(long,long);
  OXSTREAM(unsigned long,u_long);
  OXSTREAM(short,short);
  OXSTREAM(unsigned short,u_short);
  OXSTREAM(char,char);
#ifndef _CRAY		
  OXSTREAM(unsigned char,u_char);
#endif		
  OXSTREAM(float,float);
  OXSTREAM(double,double);
	
  oxstream& operator << (xbyte x) {
    if(fputc(x.byte(),buf) == EOF) set(badbit);
    return *this;
  }
 
};

class ioxstream : public oxstream {
 public:
  void open(const char *filename, open_mode mode=out) {
    xopen(filename,(mode & app) ? "a+" : ((mode & trunc) ? "w+" : 
					  ((mode & out) ? "r+" : "r")),
	  XDR_ENCODE);
    decode=false;
  }
	
  void Decode() {
    if(!decode) {
      xdrstdio_create(&xdrs, buf, XDR_DECODE);
      decode=true;
    }
  }
  
  ioxstream() {}
  ioxstream(const char *filename) {open(filename);}
  ioxstream(const char *filename, open_mode mode) {open(filename,mode);}
  virtual ~ioxstream() {close();}

  typedef ioxstream& (*imanip)(ioxstream&);
  ioxstream& operator >> (imanip func) { return (*func)(*this); }
	
  IOXSTREAM(int,int);
  IOXSTREAM(unsigned int,u_int);
  IOXSTREAM(long,long);
  IOXSTREAM(unsigned long,u_long);
  IOXSTREAM(short,short);
  IOXSTREAM(unsigned short,u_short);
  IOXSTREAM(char,char);
#ifndef _CRAY		
  IOXSTREAM(unsigned char,u_char);
#endif		
  IOXSTREAM(float,float);
  IOXSTREAM(double,double);
	
  ioxstream& operator >> (xbyte& x) {
    x=fgetc(buf);
    if(x.byte() == EOF) set(eofbit);
    return *this;
  }
};

inline oxstream& endl(oxstream& s) {s.flush(); return s;}
inline oxstream& flush(oxstream& s) {s.flush(); return s;}

#undef IXSTREAM
#undef OXSTREAM

}

#endif
