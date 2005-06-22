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

namespace run {

using vm::pop;
using vm::read;
using vm::array;
using camp::tab;

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
  checkArray(a);
  size_t size=(size_t) a->size();
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
  checkArray(a);
  size_t size=(size_t) a->size();
  array *c=new array(size);
  for(size_t i=0; i < size; i++)
      (*c)[i]=op<T>()(b,read<T>(a,i),i);
  s->push(c);
}

template<class T>
void arrayNegate(vm::stack *s)
{
  array *a=pop<array*>(s);
  checkArray(a);
  size_t size=(size_t) a->size();
  array *c=new array(size);
  for(size_t i=0; i < size; i++)
    (*c)[i]=-read<T>(a,i);
  s->push(c);
}

template<class T>
void sumArray(vm::stack *s)
{
  array *a=pop<array*>(s);
  checkArray(a);
  size_t size=(size_t) a->size();
  T sum=0;
  for(size_t i=0; i < size; i++)
    sum += read<T>(a,i);
  s->push(sum);
}

template<class T>
void maxArray(vm::stack *s)
{
  array *a=pop<array*>(s);
  checkArray(a);
  size_t size=(size_t) a->size();
  if(size == 0) vm::error("cannot take max of empty array");
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
  array *a=pop<array*>(s);
  checkArray(a);
  size_t size=(size_t) a->size();
  if(size == 0) vm::error("cannot take min of empty array");
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
    array *a=vm::get<array*>(A);
    array *b=vm::get<array*>(B);
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

template<class T>
void write(vm::stack *s)
{
  T val = pop<T>(s);
  camp::file *f = pop<camp::file*>(s);
  if(!f->isOpen()) return;
  if(f->Standard() && settings::suppressStandard) return;
  f->write(val);
}

template<class T>
void writen(vm::stack *s)
{
  T val = pop<T>(s);
  if(settings::suppressStandard) return;
  camp::Stdout.resetlines();
  camp::Stdout.write(val);
  camp::Stdout.writeline();
}

template<class T>
void write2(vm::stack *s)
{
  T val2 = pop<T>(s);
  T val1 = pop<T>(s);
  if(settings::suppressStandard) return;
  camp::Stdout.resetlines();
  camp::Stdout.write(val1);
  camp::Stdout.write(tab);
  camp::Stdout.write(val2);
  camp::Stdout.writeline();
}

template<class T>
void write3(vm::stack *s)
{
  T val3 = pop<T>(s);
  T val2 = pop<T>(s);
  T val1 = pop<T>(s);
  if(settings::suppressStandard) return;
  camp::Stdout.resetlines();
  camp::Stdout.write(val1);
  camp::Stdout.write(tab);
  camp::Stdout.write(val2);
  camp::Stdout.write(tab);
  camp::Stdout.write(val3);
  camp::Stdout.writeline();
}

// write an array to stdout, with indices
template<class T>
void showArray(vm::stack *s)
{
  array *a=pop<array*>(s);
  if(settings::suppressStandard) return;
  camp::Stdout.resetlines();
  checkArray(a);
  size_t size=(size_t) a->size();
  for(size_t i=0; i < size; i++) {
    std::cout << i << ":\t";
    camp::Stdout.write(read<T>(a,i));
    camp::Stdout.writeline();
  }
  flush(std::cout);
}

template<class T>
void writeArray(vm::stack *s)
{
  array *a=pop<array*>(s);
  camp::file *f = pop<camp::file*>(s);
  if(!f->isOpen()) return;
  if(f->Standard() && settings::suppressStandard) return;
  checkArray(a);
  size_t size=(size_t) a->size();
  for(size_t i=0; i < size; i++) {
    f->write(read<T>(a,i));
    if(f->text()) f->writeline();
  }
  f->flush();
}

template<class T>
void outArray2(camp::file *f, array *a)
{
  if(f->Standard() && settings::suppressStandard) return;
  checkArray(a);
  size_t size=(size_t) a->size();
  for(size_t i=0; i < size; i++) {
    array *ai=read<array*>(a,i);
    checkArray(ai);
    size_t aisize=(size_t) ai->size();
    for(size_t j=0; j < aisize; j++) {
      if(j > 0 && f->text()) f->write(tab);
      f->write(read<T>(ai,j));
    }
    if(f->text()) f->writeline();
  }
  f->flush();
}

template<class T>
void showArray2(vm::stack *s)
{
  array *a=pop<array*>(s);
  outArray2<T>(&camp::Stdout,a);
}

template<class T>
void writeArray2(vm::stack *s)
{
  array *a=pop<array*>(s);
  camp::file *f = pop<camp::file*>(s);
  if(!f->isOpen()) return;
  outArray2<T>(f,a);
}

template<class T>
void outArray3(camp::file *f, array *a)
{
  if(f->Standard() && settings::suppressStandard) return;
  checkArray(a);
  size_t size=(size_t) a->size();
  for(size_t i=0; i < size; i++) {
    array *ai=read<array*>(a,i);
    checkArray(ai);
    size_t aisize=(size_t) ai->size();
    for(size_t j=0; j < aisize; j++) {
      array *aij=read<array*>(ai,j);
      checkArray(aij);
      size_t aijsize=(size_t) aij->size();
      for(size_t k=0; k < aijsize; k++) {
	if(k > 0 && f->text()) f->write(tab);
	f->write(read<T>(aij,k));
      }
      if(f->text()) f->writeline();
    }
    if(f->text()) f->writeline();
  }
  f->flush();
}

template<class T>
void showArray3(vm::stack *s)
{
  array *a=pop<array*>(s);
  outArray3<T>(&camp::Stdout,a);
}

template<class T>
void writeArray3(vm::stack *s)
{
  array *a=pop<array*>(s);
  camp::file *f = pop<camp::file*>(s);
  if(!f->isOpen()) return;
  outArray3<T>(f,a);
}

template <double (*func)(double)>
void realArrayFunc(vm::stack *s) 
{
  array *a=pop<array*>(s);
  checkArray(a);
  size_t size=(size_t) a->size();
  array *c=new array(size);
  for(size_t i=0; i < size; i++) {
    double x=read<double>(a,i);
    (*c)[i]=func(x);
  }
  s->push(c);
}

} // namespace run

#endif // ARRAYOP_H
