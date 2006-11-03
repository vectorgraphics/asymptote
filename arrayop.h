/*****
 * arrayop
 * John Bowman
 *
 * Array operations
 *****/
#ifndef ARRAYOP_H
#define ARRAYOP_H

#include <typeinfo>

#include "util.h"
#include "stack.h"
#include "array.h"
#include "types.h"
#include "fileio.h"
#include "callable.h"

namespace run {

using vm::pop;
using vm::read;
using vm::array;
using camp::tab;

vm::array *copyArray(vm::array *a);
vm::array *copyArray2(vm::array *a);
  
template<class T, template <class S> class op>
void arrayArrayOp(vm::stack *s)
{
  array *b=pop<array*>(s);
  array *a=pop<array*>(s);
  size_t size=checkArrays(a,b);
  array *c=new array(size);
  for(size_t i=0; i < size; i++)
      (*c)[i]=op<T>()(read<T>(a,i),read<T>(b,i),i);
  s->push(c);
}

template<class T, template <class S> class op>
void arrayOp(vm::stack *s)
{
  T b=pop<T>(s);
  array *a=pop<array*>(s);
  size_t size=checkArray(a);
  array *c=new array(size);
  for(size_t i=0; i < size; i++)
      (*c)[i]=op<T>()(read<T>(a,i),b,i);
  s->push(c);
}

template<class T, template <class S> class op>
void opArray(vm::stack *s)
{
  array *a=pop<array*>(s);
  T b=pop<T>(s);
  size_t size=checkArray(a);
  array *c=new array(size);
  for(size_t i=0; i < size; i++)
      (*c)[i]=op<T>()(b,read<T>(a,i),i);
  s->push(c);
}

template<class T>
void arrayNegate(vm::stack *s)
{
  array *a=pop<array*>(s);
  size_t size=checkArray(a);
  array *c=new array(size);
  for(size_t i=0; i < size; i++)
    (*c)[i]=-read<T>(a,i);
  s->push(c);
}

template<class T>
void sumArray(vm::stack *s)
{
  array *a=pop<array*>(s);
  size_t size=checkArray(a);
  T sum=0;
  for(size_t i=0; i < size; i++)
    sum += read<T>(a,i);
  s->push(sum);
}

extern const char *emptyarray;
  
template<class T, template <class S> class op>
void binopArray(vm::stack *s)
{
  array *a=pop<array*>(s);
  size_t size=checkArray(a);
  if(size == 0) vm::error(emptyarray);
  T m=read<T>(a,0);
  for(size_t i=1; i < size; i++)
    m=op<T>()(m,read<T>(a,i));
  s->push(m);
}

template<class T, template <class S> class op>
void binopArray2(vm::stack *s)
{
  array *a=pop<array*>(s);
  size_t size=checkArray(a);
  bool empty=true;
  T m=0;
  for(size_t i=0; i < size; i++) {
    array *ai=read<array*>(a,i);
    size_t aisize=checkArray(ai);
    if(aisize) {
      if(empty) {
	m=read<T>(ai,0);
	empty=false;
      }
      for(size_t j=0; j < aisize; j++)
	m=op<T>()(m,read<T>(ai,j));
    }
  }
  if(empty) vm::error(emptyarray);
  s->push(m);
}

template<class T, template <class S> class op>
void binopArray3(vm::stack *s)
{
  array *a=pop<array*>(s);
  size_t size=checkArray(a);
  bool empty=true;
  T m=0;
  for(size_t i=0; i < size; i++) {
    array *ai=read<array*>(a,i);
    size_t aisize=checkArray(ai);
    for(size_t j=0; j < aisize; j++) {
      array *aij=read<array*>(ai,j);
      size_t aijsize=checkArray(aij);
      if(aijsize) {
	if(empty) {
	  m=read<T>(aij,0);
	  empty=false;
	}
	for(size_t k=0; k < aijsize; k++) {
	  m=op<T>()(m,read<T>(aij,k));
	}
      }
    }
  }
  if(empty) vm::error(emptyarray);
  s->push(m);
}

template<class T>
struct compare {
  bool operator() (const vm::item& a, const vm::item& b)
  {
    return vm::get<T>(a) < vm::get<T>(b);
  }
};

template<class T>
void sortArray(vm::stack *s)
{
  array *c=copyArray(pop<array*>(s));
  sort(c->begin(),c->end(),compare<T>());
  s->push(c);
}

template<class T>
struct compare2 {
  bool operator() (const vm::item& A, const vm::item& B)
  {
    array *a=vm::get<array*>(A);
    array *b=vm::get<array*>(B);
    size_t size=a->size();
    if(size != b->size()) return false;

    for(size_t j=0; j < size; j++) {
      if(read<T>(a,j) < read<T>(b,j)) return true;
      if(read<T>(a,j) > read<T>(b,j)) return false;
    }
    return false;
  }
};

// Sort the rows of a 2-dimensional array by the first column, breaking
// ties with successively higher columns.
template<class T>
void sortArray2(vm::stack *s)
{
  array *c=copyArray(pop<array*>(s));
  stable_sort(c->begin(),c->end(),compare2<T>());
  s->push(c);
}

// Search a sorted ordered array of n elements to find an interval containing
// a given key. Returns n-1 if the key is greater than or equal to the last
// element, -1 if the key is less than the first element, and otherwise the
// index corresponding to the left-hand endpoint of the matching interval. 
template<class T>
void searchArray(vm::stack *s)
{
  T key=pop<T>(s);
  array *a=pop<array*>(s);
  int size=(int) a->size();
  if(size == 0) {s->push(0); return;}
  if(key < read<T>(a,0)) {s->push(-1); return;}
  int u=size-1;
  if(key >= read<T>(a,u)) {s->push(u); return;}
  int l=0;
	
  while (l < u) {
    int i=(l+u)/2;
    if(read<T>(a,i) <= key && key < read<T>(a,i+1)) {s->push(i); return;}
    if(key < read<T>(a,i)) u=i;
    else l=i+1;
  }
  s->push(0);
}

extern mem::string emptystring;
  
void writestring(vm::stack *s);
  
template<class T>
void write(vm::stack *s)
{
  array *a=pop<array*>(s);
  vm::callable *suffix=pop<vm::callable *>(s,NULL);
  T first=pop<T>(s);
  mem::string S=pop<mem::string>(s,emptystring);
  vm::item it=pop(s);
  bool defaultfile=isdefault(it);
  camp::file *f=defaultfile ? &camp::Stdout : vm::get<camp::file*>(it);
  
  size_t size=checkArray(a);
  if(!f->isOpen()) return;
  if(S != "") f->write(S);
  f->write(first);
  for(size_t i=0; i < size; ++i) {
    f->write(tab);
    f->write(read<T>(a,i));
  }
  if(f->text()) {
    if(suffix) {
      s->push(f);
      suffix->call(s);
    } else if(defaultfile) {
      try {
	f->writeline();
      } catch (quit&) {
      }
    }
  }
}

template<class T>
void writeArray(vm::stack *s)
{
  array *A=pop<array*>(s);
  array *a=pop<array*>(s);
  mem::string S=pop<mem::string>(s,emptystring);
  vm::item it=pop(s);
  bool defaultfile=isdefault(it);
  camp::file *f=defaultfile ? &camp::Stdout : vm::get<camp::file*>(it);
  
  size_t asize=checkArray(a);
  size_t Asize=checkArray(A);
  if(f->Standard()) camp::Stdout.resetlines();
  else if(!f->isOpen()) return;
  try {
    if(S != "") {f->write(S); f->writeline();}
  
    size_t i=0;
    bool cont=true;
    while(cont) {
      cont=false;
      bool first=true;
      if(i < asize) {
	if(defaultfile) std::cout << i << ":\t";
	f->write(read<T>(a,i)); cont=true;
	first=false;
      }
      unsigned count=0;
      for(size_t j=0; j < Asize; ++j) {
	array *Aj=read<array*>(A,j);
	size_t Ajsize=checkArray(Aj);
	if(i < Ajsize) {
	  if(f->text()) {
	    if(first && defaultfile) std::cout << i << ":\t";
	    for(unsigned k=0; k <= count; ++k)
	      f->write(tab);
	    count=0;
	  }
	  f->write(read<T>(Aj,i));
	  first=false;
	  cont=true;
	} else count++;
      }
      ++i;
      if(cont && f->text()) f->writeline();
    }
  } catch (quit&) {
  }
  f->flush();
}
  
template<class T>
void writeArray2(vm::stack *s)
{
  array *a=pop<array*>(s);
  camp::file *f=pop<camp::file*>(s,&camp::Stdout);
  
  size_t size=checkArray(a);
  if(f->Standard()) camp::Stdout.resetlines();
  else if(!f->isOpen()) return;
  
  try {
  for(size_t i=0; i < size; i++) {
    array *ai=read<array*>(a,i);
    size_t aisize=checkArray(ai);
    for(size_t j=0; j < aisize; j++) {
      if(j > 0 && f->text()) f->write(tab);
      f->write(read<T>(ai,j));
    }
    if(f->text()) f->writeline();
  }
  } catch (quit&) {
  }
  f->flush();
}

template<class T>
void writeArray3(vm::stack *s)
{
  array *a=pop<array*>(s);
  camp::file *f=pop<camp::file*>(s,&camp::Stdout);
  
  size_t size=checkArray(a);
  if(f->Standard()) camp::Stdout.resetlines();
  else if(!f->isOpen()) return;
  
  try {
  for(size_t i=0; i < size;) {
    array *ai=read<array*>(a,i);
    size_t aisize=checkArray(ai);
    for(size_t j=0; j < aisize; j++) {
      array *aij=read<array*>(ai,j);
      size_t aijsize=checkArray(aij);
      for(size_t k=0; k < aijsize; k++) {
	if(k > 0 && f->text()) f->write(tab);
	f->write(read<T>(aij,k));
      }
      if(f->text()) f->writeline();
    }
    ++i;
    if(i < size && f->text()) f->writeline();
  }
  } catch (quit&) {
  }
  f->flush();
}

template <double (*func)(double)>
void realArrayFunc(vm::stack *s) 
{
  array *a=pop<array*>(s);
  size_t size=checkArray(a);
  array *c=new array(size);
  for(size_t i=0; i < size; i++) {
    double x=read<double>(a,i);
    (*c)[i]=func(x);
  }
  s->push(c);
}

} // namespace run

#endif // ARRAYOP_H
