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
#include "inst.h"
#include "types.h"
#include "fileio.h"
#include "builtin.h"

namespace run {

using vm::pop;
using vm::read;
using vm::array;
using camp::tab;
using camp::newline;
  
template<class T, template <class S> class op>
void arrayArrayOp(vm::stack *s)
{
  array *b=pop<array *>(s);
  array *a=pop<array *>(s);
  size_t size=checkArrays(s,a,b);
  array *c=new array(size);
  for(size_t i=0; i < size; i++)
      (*c)[i]=op<T>()(read<T>(a,i),read<T>(b,i),s,i);
  s->push(c);
}

template<class T, template <class S> class op>
void arrayOp(vm::stack *s)
{
  T b=s->pop<T>();
  array *a=pop<array *>(s);
  checkArray(s,a);
  size_t size=(size_t) a->size();
  array *c=new array(size);
  for(size_t i=0; i < size; i++)
      (*c)[i]=op<T>()(read<T>(a,i),b,s,i);
  s->push(c);
}

template<class T, template <class S> class op>
void opArray(vm::stack *s)
{
  array *a=pop<array *>(s);
  T b=s->pop<T>();
  checkArray(s,a);
  size_t size=(size_t) a->size();
  array *c=new array(size);
  for(size_t i=0; i < size; i++)
      (*c)[i]=op<T>()(b,read<T>(a,i),s,i);
  s->push(c);
}

template<class T>
void arrayNegate(vm::stack *s)
{
  array *a=pop<array *>(s);
  checkArray(s,a);
  size_t size=(size_t) a->size();
  array *c=new array(size);
  for(size_t i=0; i < size; i++)
      (*c)[i]=-read<T>(a,i);
  s->push(c);
}

template<class T>
void sumArray(vm::stack *s)
{
  array *a=pop<array *>(s);
  checkArray(s,a);
  size_t size=(size_t) a->size();
  T sum=0;
  for(size_t i=0; i < size; i++)
    sum += read<T>(a,i);
  s->push(sum);
}

template<class T>
void maxArray(vm::stack *s)
{
  array *a=pop<array *>(s);
  checkArray(s,a);
  size_t size=(size_t) a->size();
  if(size == 0) error(s,"cannot take max of empty array");
  T m=read<T>(a,0);
  for(size_t i=1; i < size; i++) {
    T val=read<T>(a,i);
    if(val > m) m=val;
  }
  s->push(m);
}

template<class T>
void minArray(vm::stack *s)
{
  array *a=pop<array *>(s);
  checkArray(s,a);
  size_t size=(size_t) a->size();
  if(size == 0) error(s,"cannot take min of empty array");
  T m=read<T>(a,0);
  for(size_t i=1; i < size; i++) {
    T val=read<T>(a,i);
    if(val < m) m=val;
  }
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
  array *c=copyArray(s);
  sort(c->begin(),c->end(),compare<T>());
  s->push(c);
}

template<class T>
struct compare2 {
  bool operator() (const vm::item& A, const vm::item& B)
  {
    array *a=vm::get<array *>(A);
    array *b=vm::get<array *>(B);
    size_t size=(size_t) a->size();
    if(size != (size_t) b->size()) return false;

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
  array *c=copyArray(s);
  stable_sort(c->begin(),c->end(),compare2<T>());
  s->push(c);
}

template<class T>
void write(vm::stack *s)
{
  T val = s->pop<T>();
  camp::file *f = s->pop<camp::file*>();
  if(!f->isOpen()) return;
  if(f->standard() && settings::suppressStandard) return;
  f->write(val);
}

template<class T>
void writen(vm::stack *s)
{
  T val = s->pop<T>();
  if(settings::suppressStandard) return;
  camp::Stdout.write(val);
  camp::Stdout.write(newline);
}

template<class T>
void write2(vm::stack *s)
{
  T val2 = s->pop<T>();
  T val1 = s->pop<T>();
  if(settings::suppressStandard) return;
  camp::Stdout.write(val1);
  camp::Stdout.write(tab);
  camp::Stdout.write(val2);
  camp::Stdout.write(newline);
}

template<class T>
void write3(vm::stack *s)
{
  T val3 = s->pop<T>();
  T val2 = s->pop<T>();
  T val1 = s->pop<T>();
  if(settings::suppressStandard) return;
  camp::Stdout.write(val1);
  camp::Stdout.write(tab);
  camp::Stdout.write(val2);
  camp::Stdout.write(tab);
  camp::Stdout.write(val3);
  camp::Stdout.write(newline);
}

template<class T>
void writeP(vm::stack *s)
{
  const T& val = *(s->pop<T*>());
  camp::file *f = s->pop<camp::file*>();
  if(!f->isOpen()) return;
  if(f->standard() && settings::suppressStandard) return;
  f->write(val);
}
  
// write an array to stdout, with indices
template<class T>
void showArray(vm::stack *s)
{
  array *a=pop<array *>(s);
  if(settings::suppressStandard) return;
  checkArray(s,a);
  size_t size=(size_t) a->size();
  for(size_t i=0; i < size; i++) {
    cout << i << ":\t";
    camp::Stdout.write(read<T>(a,i));
    cout << "\n";
  }
  flush(cout);
}

template<class T>
void writeArray(vm::stack *s)
{
  array *a=pop<array *>(s);
  camp::file *f = s->pop<camp::file*>();
  if(!f->isOpen()) return;
  if(f->standard() && settings::suppressStandard) return;
  checkArray(s,a);
  size_t size=(size_t) a->size();
  for(size_t i=0; i < size; i++) {
    f->write(read<T>(a,i));
    if(f->text()) f->write(newline);
  }
  f->flush();
}

template<class T>
void outArray2(vm::stack *s, camp::file *f, array *a)
{
  if(f->standard() && settings::suppressStandard) return;
  checkArray(s,a);
  size_t size=(size_t) a->size();
  for(size_t i=0; i < size; i++) {
    array *ai=read<array *>(a,i);
    checkArray(s,ai);
    size_t aisize=(size_t) ai->size();
    for(size_t j=0; j < aisize; j++) {
      if(j > 0 && f->text()) f->write(tab);
      f->write(read<T>(ai,j));
    }
    if(f->text()) f->write(newline);
  }
  f->flush();
}

template<class T>
void showArray2(vm::stack *s)
{
  array *a=pop<array *>(s);
  outArray2<T>(s,&camp::Stdout,a);
}

template<class T>
void writeArray2(vm::stack *s)
{
  array *a=pop<array *>(s);
  camp::file *f = s->pop<camp::file*>();
  if(!f->isOpen()) return;
  outArray2<T>(s,f,a);
}

template<class T>
void writeArray3(vm::stack *s)
{
  array *a=pop<array *>(s);
  camp::file *f = s->pop<camp::file*>();
  if(!f->isOpen()) return;
  if(f->standard() && settings::suppressStandard) return;
  checkArray(s,a);
  size_t size=(size_t) a->size();
  for(size_t i=0; i < size; i++) {
    array *ai=read<array *>(a,i);
    checkArray(s,ai);
    size_t aisize=(size_t) ai->size();
    for(size_t j=0; j < aisize; j++) {
      array *aij=read<array *>(ai,j);
      checkArray(s,aij);
      size_t aijsize=(size_t) aij->size();
      for(size_t k=0; k < aijsize; k++) {
	if(k > 0 && f->text()) f->write(tab);
	f->write(read<T>(aij,k));
      }
      if(f->text()) f->write(newline);
    }
    if(f->text()) f->write(newline);
  }
  f->flush();
}

template <double (*func)(double)>
void realArrayFunc(vm::stack *s) 
{
  array *a=pop<array *>(s);
  checkArray(s,a);
  size_t size=(size_t) a->size();
  array *c=new array(size);
  for(size_t i=0; i < size; i++)
    (*c)[i]=func(read<double>(a,i));
  s->push(c);
}

template<class T, template <class S> class op>
void binaryOp(vm::stack *s)
{
  T w = pop<T>(s);
  T z = pop<T>(s);
  s->push(op<T>()(z,w,s,0));
}

} // namespace run

#endif // ARRAYOP_H
