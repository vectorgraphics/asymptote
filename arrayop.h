/*****
 * arrayop
 * John Bowman
 *
 * Array operations
 *****/
#ifndef ARRAYOP_H
#define ARRAYOP_H

#include "util.h"
#include "stack.h"
#include "array.h"
#include "types.h"
#include "fileio.h"
#include "callable.h"
#include "mathop.h"
#include "errormsg.h"

namespace run {

using vm::pop;
using vm::read;
using vm::array;
using camp::tab;

vm::array *copyArray(vm::array *a);
vm::array *copyArray2(vm::array *a);

// Call the write(file, suffix) method on a record.
// frame is the record's vmFrame, t is the record type (ty with kind ty_record),
// f is the target file.
void callRecordWriteMethod(vm::stack *s, vm::vmFrame *frame, types::ty *t, camp::file *f);

// Tests whether t is the type of a write suffix, void (file).
bool isWriteSuffixType(types::ty *t);

template<class T, class U, template <class S> class op>
void arrayOp(vm::stack *s)
{
  U b=pop<U>(s);
  array *a=pop<array*>(s);
  size_t size=checkArray(a);
  array *c=new array(size);
  for(size_t i=0; i < size; i++)
    (*c)[i]=op<T>()(read<T>(a,i),b,i);
  s->push(c);
}

template<class T, class U, template <class S> class op>
void opArray(vm::stack *s)
{
  array *a=pop<array*>(s);
  T b=pop<T>(s);
  size_t size=checkArray(a);
  array *c=new array(size);
  for(size_t i=0; i < size; i++)
    (*c)[i]=op<U>()(b,read<U>(a,i),i);
  s->push(c);
}

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

extern const char *arrayempty;

template<class T, template <class S> class op>
void binopArray(vm::stack *s)
{
  array *a=pop<array*>(s);
  size_t size=checkArray(a);
  if(size == 0) vm::error(arrayempty);
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
  T m;
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
  if(!empty)
    s->push(m);
  else
    vm::error(arrayempty);
}

template<class T, template <class S> class op>
void binopArray3(vm::stack *s)
{
  array *a=pop<array*>(s);
  size_t size=checkArray(a);
  bool empty=true;
  T m;
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
  if(!empty)
    s->push(m);
  else
    vm::error(arrayempty);
}

template<class T, class U, template <class S> class op>
void array2Op(vm::stack *s)
{
  U b=pop<U>(s);
  array *a=pop<array*>(s);
  size_t size=checkArray(a);
  array *c=new array(size);
  for(size_t i=0; i < size; ++i) {
    array *ai=read<array*>(a,i);
    size_t aisize=checkArray(ai);
    array *ci=new array(aisize);
    (*c)[i]=ci;
    for(size_t j=0; j < aisize; j++)
      (*ci)[j]=op<T>()(read<T>(ai,j),b,0);
  }
  s->push(c);
}

template<class T, class U, template <class S> class op>
void opArray2(vm::stack *s)
{
  array *a=pop<array*>(s);
  T b=pop<T>(s);
  size_t size=checkArray(a);
  array *c=new array(size);
  for(size_t i=0; i < size; ++i) {
    array *ai=read<array*>(a,i);
    size_t aisize=checkArray(ai);
    array *ci=new array(aisize);
    (*c)[i]=ci;
    for(size_t j=0; j < aisize; j++)
      (*ci)[j]=op<U>()(read<U>(ai,j),b,0);
  }
  s->push(c);
}

template<class T, template <class S> class op>
void array2Array2Op(vm::stack *s)
{
  array *b=pop<array*>(s);
  array *a=pop<array*>(s);
  size_t size=checkArrays(a,b);
  array *c=new array(size);
  for(size_t i=0; i < size; ++i) {
    array *ai=read<array*>(a,i);
    array *bi=read<array*>(b,i);
    size_t aisize=checkArrays(ai,bi);
    array *ci=new array(aisize);
    (*c)[i]=ci;
    for(size_t j=0; j < aisize; j++)
      (*ci)[j]=op<T>()(read<T>(ai,j),read<T>(bi,j),0);
  }
  s->push(c);
}

template<class T>
bool Array2Equals(vm::stack *s)
{
  array *b=pop<array*>(s);
  array *a=pop<array*>(s);
  size_t n=checkArray(a);
  if(n != checkArray(b)) return false;
  if(n == 0) return true;
  size_t n0=checkArray(read<array*>(a,0));
  if(n0 != checkArray(read<array*>(b,0))) return false;

  for(size_t i=0; i < n; ++i) {
    array *ai=read<array*>(a,i);
    array *bi=read<array*>(b,i);
    for(size_t j=0; j < n0; ++j) {
      if(read<T>(ai,j) != read<T>(bi,j))
        return false;
    }
  }
  return true;
}

template<class T>
void array2Equals(vm::stack *s)
{
  s->push(Array2Equals<T>(s));
}

template<class T>
void array2NotEquals(vm::stack *s)
{
  s->push(!Array2Equals<T>(s));
}

template<class T>
void diagonal(vm::stack *s)
{
  array *a=pop<array*>(s);
  size_t n=checkArray(a);
  array *c=new array(n);
  for(size_t i=0; i < n; ++i) {
    array *ci=new array(n);
    (*c)[i]=ci;
    for(size_t j=0; j < i; ++j)
      (*ci)[j]=T();
    (*ci)[i]=read<T>(a,i);
    for(size_t j=i+1; j < n; ++j)
      (*ci)[j]=T();
  }
  s->push(c);
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

// Search a sorted ordered array a of n elements for key, returning the index i
// if a[i] <= key < a[i+1], -1 if key is less than all elements of a, or
// n-1 if key is greater than or equal to the last element of a.
template<class T>
void searchArray(vm::stack *s)
{
  T key=pop<T>(s);
  array *a=pop<array*>(s);
  size_t size= a->size();
  if(size == 0 || key < read<T>(a,0)) {s->push(-1); return;}
  size_t u=size-1;
  if(key >= read<T>(a,u)) {s->push((Int) u); return;}
  size_t l=0;

  while (l < u) {
    size_t i=(l+u)/2;
    if(key < read<T>(a,i)) u=i;
    else if(key < read<T>(a,i+1)) {s->push((Int) i); return;}
    else l=i+1;
  }
  s->push(0);
}

extern string emptystring;

void writestring(vm::stack *s);

// Generic write fallback for heterogeneous arguments.
// Receives a single array of tagged_var* elements, built by the
// transHeteroWrite handler (builtin_handlers.cc).
// Each element is a tagged_var whose ->tag is the full types::ty * pointer
// (stored as an Int) and whose ->value holds the actual vm::item.
// Scans the elements to identify the file (first, if ty_file), label
// (next, if ty_string), optional suffix (last, if of type void (file)),
// and data values (everything else).
inline void write_var(vm::stack *s)
{
  array *arr = pop<array *>(s);
  size_t n = checkArray(arr);


  // Helper: resolve the effective type and value for element i.
  // The tag in tagged_var is the types::ty * pointer as an Int.  Every
  // element is keyed by a concrete type: the transHeteroWrite handler wraps
  // only values it has validated as writeable.
  struct tv_res {
    types::ty *t;
    vm::item value;
  };
  auto getTV = [&](size_t i) -> tv_res {
    vm::tagged_var *tv = vm::get<vm::tagged_var *>((*arr)[i]);
    return { (types::ty *)(intptr_t)tv->tag, tv->value };
  };

  camp::file *f = &camp::Stdout;
  bool defaultfile = true;
  string label;
  bool haveLabel = false;
  size_t i = 0;

  // Consume optional file (first element, if file type).
  if (i < n && getTV(i).t->kind == types::ty_file) {
    vm::item vi = getTV(i).value;
    f = isdefault(vi) ? &camp::Stdout : vm::get<camp::file *>(vi);
    defaultfile = isdefault(vi);
    ++i;
  }

  // Consume optional label (next element, if string).
  if (i < n && getTV(i).t->kind == types::ty_string) {
    label = vm::get<string>(getTV(i).value);
    haveLabel = true;
    ++i;
  }

  // Check for suffix (last element, if of type void (file)).  The test must
  // be on the full type recorded in the tag, not just its kind: the callable
  // is handed a file and expected to return nothing, so calling a function of
  // any other signature would corrupt the stack.
  vm::callable *suffix = NULL;
  size_t dataEnd = n;
  if (n > 0) {
    size_t last = n - 1;
    if (last >= i) {
      tv_res lastTV = getTV(last);
      if (isWriteSuffixType(lastTV.t)) {
        suffix = vm::get<vm::callable *>(lastTV.value);
        dataEnd = n - 1;
      }
    }
  }

  if (!f->isOpen() || !f->enabled()) return;

  if (f->Standard()) interact::lines = 0;

  if (haveLabel && label != "") f->write(label);

  // Write one data value to f.  Scalars are written directly.  Arrays of
  // any depth are written recursively: tab between elements on the same
  // line, newline between lines, and (depth-2) blank lines between blocks
  // at each level above the innermost -- matching the type-specific
  // write(file, array) builtins.  Records, whether standalone or the cells
  // of an array, are written by calling their write(file, suffix) method.
  // A type that cannot be written is a runtime error.
  bool firstWritten = true;
  auto beginValue = [&]() {
    if (!firstWritten) f->write(tab);
    firstWritten = false;
  };
  auto writeScalar = [](camp::file *f, types::ty *t, vm::item val) -> bool {
    switch (t->kind) {
      case types::ty_boolean:   f->write(vm::get<bool>(val)); return true;
      case types::ty_Int:       f->write(vm::get<Int>(val)); return true;
      case types::ty_real:      f->write(vm::get<double>(val)); return true;
      case types::ty_pair:      f->write(vm::get<camp::pair>(val)); return true;
      case types::ty_triple:    f->write(vm::get<camp::triple>(val)); return true;
      case types::ty_string:    f->write(vm::get<string>(val)); return true;
      case types::ty_pen:       f->write(vm::get<camp::pen>(val)); return true;
      case types::ty_guide:
                               f->write(vm::get<camp::guide *>(val)); return true;
      case types::ty_path:
                               f->write(new camp::pathguide(*vm::get<camp::path *>(val))); return true;
      case types::ty_transform: f->write(vm::get<camp::transform>(val)); return true;
      default: return false;
    }
  };
  auto writeRecord = [&](types::ty *t, vm::item val) {
    vm::vmFrame *recFrame = vm::get<vm::vmFrame *>(val);
    if (!recFrame)
      vm::error("dereference of null pointer");
    callRecordWriteMethod(s, recFrame, t, f);
  };
  auto writeArr = [&](auto&& self, types::ty *elemTy, vm::array *a, int depth, int totalDepth) -> void {
    size_t n = checkArray(a);
    for (size_t k = 0; k < n; ++k) {
      vm::item &it = (*a)[k];
      if (it.empty()) continue;
      if (depth == 1) {
        if (elemTy->kind == types::ty_record)
          writeRecord(elemTy, it);
        else if (!writeScalar(f, elemTy, it)) {
          ostringstream msg;
          msg << "cannot write value of type '" << *elemTy << "'";
          vm::error(msg);
        }
      } else {
        vm::array *sub = vm::get<vm::array *>(it);
        self(self, elemTy, sub, depth - 1, totalDepth);
      }
      if (k + 1 < n && f->text()) {
        if (depth == 1 && totalDepth > 1)
          f->write(tab);
        else if (depth == 1)
          f->writeline();
        else {
          f->writeline();
          for (int b = 1; b < depth - 1; ++b)
            f->writeline();
        }
      }
    }
  };
  auto writeOne = [&](tv_res tv) {
    if (tv.t->kind == types::ty_array) {
      types::array *arrTy = dynamic_cast<types::array *>(tv.t);
      if (arrTy) {
        if (!firstWritten) f->writeline();
        firstWritten = false;
        vm::array *data = vm::get<vm::array *>(tv.value);
        types::ty *innerTy = arrTy->celltype;
        while (innerTy->kind == types::ty_array)
          innerTy = ((types::array *)innerTy)->celltype;
        writeArr(writeArr, innerTy, data, arrTy->depth(), arrTy->depth());
        return;
      }
    }
    beginValue();
    if (tv.t->kind == types::ty_record) {
      writeRecord(tv.t, tv.value);
      return;
    }
    if (!writeScalar(f, tv.t, tv.value)) {
      ostringstream msg;
      msg << "cannot write value of type '" << *tv.t << "'";
      vm::error(msg);
    }
  };

  try {
    for (size_t d = 0; d < dataEnd - i; ++d)
      writeOne(getTV(i + d));
  } catch (quit&) {
  }

  if (f->text()) {
    if (suffix) {
      s->push(f);
      suffix->call(s);
    } else if (defaultfile) {
      try { f->writeline(); } catch (quit&) {}
    }
  }
}


template<class T>
void writeArray(vm::stack *s)
{
  array *A=pop<array*>(s);
  array *a=pop<array*>(s);
  string S=pop<string>(s,emptystring);
  vm::item it=pop(s);
  bool defaultfile=isdefault(it);
  camp::file *f=defaultfile ? &camp::Stdout : vm::get<camp::file*>(it);
  if(!f->isOpen() || !f->enabled()) return;

  size_t asize=checkArray(a);
  size_t Asize=checkArray(A);
  if(f->Standard()) interact::lines=0;
  else if(!f->isOpen()) return;
  try {
    if(S != "") {f->write(S); f->writeline();}

    size_t i=0;
    bool cont=true;
    while(cont) {
      cont=false;
      bool first=true;
      if(i < asize) {
        vm::item& I=(*a)[i];
        if(defaultfile) cout << i << ":\t";
        if(!I.empty())
          f->write(vm::get<T>(I));
        cont=true;
        first=false;
      }
      unsigned count=0;
      for(size_t j=0; j < Asize; ++j) {
        array *Aj=read<array*>(A,j);
        size_t Ajsize=checkArray(Aj);
        if(i < Ajsize) {
          if(f->text()) {
            if(first && defaultfile) cout << i << ":\t";
            for(unsigned k=0; k <= count; ++k)
              f->write(tab);
            count=0;
          }
          vm::item& I=(*Aj)[i];
          if(!I.empty())
            f->write(vm::get<T>(I));
          cont=true;
          first=false;
        } else count++;
      }
      ++i;
      if(cont && f->text()) f->writeline();
    }
  } catch (quit&) {
  }
  f->flush();
}

template <class T, class S, T (*func)(S)>
void arrayFunc(vm::stack *s)
{
  array *a=pop<array*>(s);
  size_t size=checkArray(a);
  array *c=new array(size);
  for(size_t i=0; i < size; i++)
    (*c)[i]=func(read<S>(a,i));
  s->push(c);
}

template <class T, class S, T (*func)(S)>
void arrayFunc2(vm::stack *s)
{
  array *a=pop<array*>(s);
  size_t size=checkArray(a);
  array *c=new array(size);
  for(size_t i=0; i < size; ++i) {
    array *ai=read<array*>(a,i);
    size_t aisize=checkArray(ai);
    array *ci=new array(aisize);
    (*c)[i]=ci;
    for(size_t j=0; j < aisize; j++)
      (*ci)[j]=func(read<S>(ai,j));
  }
  s->push(c);
}

vm::array *Identity(Int n);
camp::triple operator *(const vm::array& a, const camp::triple& v);
double norm(double *a, size_t n);
double norm(camp::triple *a, size_t n);

inline size_t checkdimension(const vm::array *a, size_t dim)
{
  size_t size=checkArray(a);
  if(dim && size != dim) {
    ostringstream buf;
    buf << "array of length " << dim << " expected";
    vm::error(buf);
  }
  return size;
}

template<class T>
inline void copyArrayC(T* &dest, const vm::array *a, size_t dim=0,
                       GCPlacement placement=NoGC)
{
  size_t size=checkdimension(a,dim);
  dest=(placement == NoGC) ? new T[size] : new(placement) T[size];
  for(size_t i=0; i < size; i++)
    dest[i]=vm::read<T>(a,i);
}

template<class T, class A>
inline void copyArrayC(T* &dest, const vm::array *a, T (*cast)(A),
                       size_t dim=0, GCPlacement placement=NoGC)
{
  size_t size=checkdimension(a,dim);
  dest=(placement == NoGC) ? new T[size] : new(placement) T[size];
  for(size_t i=0; i < size; i++)
    dest[i]=cast(vm::read<A>(a,i));
}

template<typename T>
inline vm::array* copyCArray(const size_t n, const T* p)
{
  vm::array* a = new vm::array(n);
  for(size_t i=0; i < n; ++i) (*a)[i] = p[i];
  return a;
}

template<class T>
inline void copyArray2C(T* &dest, const vm::array *a, bool square=true,
                        size_t dim2=0, GCPlacement placement=NoGC)
{
  size_t n=checkArray(a);
  size_t m=(square || n == 0) ? n : checkArray(vm::read<vm::array*>(a,0));
  if(n > 0 && dim2 && m != dim2) {
    ostringstream buf;
    buf << "second matrix dimension must be " << dim2;
    vm::error(buf);
  }

  dest=(placement == NoGC) ? new T[n*m] : new(placement) T[n*m];
  for(size_t i=0; i < n; i++) {
    vm::array *ai=vm::read<vm::array*>(a,i);
    size_t aisize=checkArray(ai);
    if(aisize == m) {
      T *desti=dest+i*m;
      for(size_t j=0; j < m; j++)
        desti[j]=vm::read<T>(ai,j);
    } else
      vm::error(square ? "matrix must be square" :
                "matrix must be rectangular");
  }
}

template<typename T>
inline vm::array* copyCArray2(const size_t n, const size_t m, const T* p)
{
  vm::array* a=new vm::array(n);
  for(size_t i=0; i < n; ++i) {
    array *ai=new array(m);
    (*a)[i]=ai;
    for(size_t j=0; j < m; ++j)
      (*ai)[j]=p[m*i+j];
  }
  return a;
}

} // namespace run

#endif // ARRAYOP_H
