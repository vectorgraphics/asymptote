/*****
 * castop.h
 * Tom Prince 2005/3/18
 *
 * Defines some runtime functions used by the stack machine.
 *
 *****/

#ifndef CASTOP_H
#define CASTOP_H

#include <cfloat>

#include "common.h"
#include "stack.h"
#include "fileio.h"
#include "lexical.h"
#include "mathop.h"
#include "array.h"

namespace run {

using vm::read;
using vm::pop;

template<class T, class S>
void cast(vm::stack *s)
{
  s->push((S) pop<T>(s));
}

void castDoubleInt(vm::stack *s)
{
  double x=pop<double>(s);
  checkint(x,0);
  s->push((int) x);
}

template<class T>
void stringCast(vm::stack *s)
{
  ostringstream buf;
  buf.precision(DBL_DIG);
  buf << pop<T>(s);
  s->push((string) buf.str());
}

template<class T>
void castString(vm::stack *s)
{
  try {
    string *S=pop<string*>(s);
    if(S->empty()) {
      T x=0;
      s->push(x);
    } else s->push(lexical::cast<T>(*S));
  } catch (lexical::bad_cast&) {
    vm::error("invalid cast.");
  }
}

template<class T, class S>
void arrayToArray(vm::stack *s)
{
  vm::array *a = pop<vm::array*>(s);
  checkArray(a);
  unsigned int size=(unsigned int) a->size();
  vm::array *c=new vm::array(size);
  for(unsigned i=0; i < size; i++)
    (*c)[i]=(S) read<T>(a,i);
  s->push(c);
}

template<class T>
void read(vm::stack *s)
{
  camp::file *f = pop<camp::file*>(s);
  T val=T();
  if(f->isOpen()) {
    f->read(val);
    if(f->LineMode()) f->nexteol();
  }
  s->push(val);
}

inline int Limit(int nx) {return nx == 0 ? INT_MAX : nx;}
inline void reportEof(camp::file *f, int count) 
{
  ostringstream buf;
  buf << "EOF after reading " << count
      << " values from file '" << f->filename() << "'.";
  vm::error(buf);
}

template<class T>
void readArray(vm::stack *s)
{
  camp::file *f = pop<camp::file*>(s);
  vm::array *c=new vm::array(0);
  if(f->isOpen()) {
    int nx=f->Nx();
    if(nx == -2) {f->read(nx); if(nx == 0) {s->push(c); return;}}
    int ny=f->Ny();
    if(ny == -2) {f->read(ny); if(ny == 0) {s->push(c); return;}}
    int nz=f->Nz();
    if(nz == -2) {f->read(nz); if(nz == 0) {s->push(c); return;}}
    T v;
    if(nx >= 0) {
      for(int i=0; i < Limit(nx); i++) {
	if(ny >= 0) {
	  vm::array *ci=new vm::array(0);
	  for(int j=0; j < Limit(ny); j++) {
	    if(nz >= 0) {
	      vm::array *cij=new vm::array(0);
	      bool break2=false;
	      for(int k=0; k < Limit(nz); k++) {
		f->read(v);
		if(f->error()) {
		  if(nx && ny && nz) reportEof(f,(i*ny+j)*nz+k);
		  s->push(c);
		  return;
		}
		if(k == 0) {
		  if(j == 0) c->push(ci);
		  ci->push(cij);
		}
		cij->push(v);
		if(f->LineMode() && f->nexteol()) {
		  if(f->nexteol()) break2=true;
		  break;
		}
	      }
	      if(break2) break;
	    } else {
	      f->read(v);
	      if(f->error()) {
		if(nx && ny) reportEof(f,i*ny+j);
		s->push(c);
		return;
	      }
	      if(j == 0) c->push(ci);
	      ci->push(v);
	      if(f->LineMode() && f->nexteol()) break;
	    }
	  }
	} else {
	  f->read(v);
	  if(f->error()) {
	    if(nx) reportEof(f,i);
	    s->push(c);
	    return;
	  }
	  c->push(v);
	  if(f->LineMode() && f->nexteol()) break;
	}
      }
    } else {
      for(;;) {
	f->read(v);
	if(f->error()) break;
	c->push(v);
	if(f->LineMode() && f->nexteol()) break;
      }
    }
  }
  s->push(c);
}

} // namespace run

#endif // CASTOP_H

