/*****
 * builtin.h
 * Tom Prince 2004/08/25
 *
 * Initialize builtins.
 *****/
#ifndef BUILTIN_H
#define BUILTIN_H

#include <typeinfo>

#include "util.h"
#include "stack.h"
#include "types.h"
#include "fileio.h"
#include "pow.h"

namespace trans {

class tenv;
class venv;
class menv;

// The base environments for built-in types and functions
void base_tenv(tenv &);
void base_venv(venv &);
void base_menv(menv&);

template <typename T>
struct less {
  bool operator() (T x, T y, vm::stack *, size_t=0) {return x < y;}
};

template <typename T>
struct lessequals {
  bool operator() (T x, T y, vm::stack *, size_t=0) {return x <= y;}
};

template <typename T>
struct equals {
  bool operator() (T x, T y, vm::stack *, size_t=0) {return x == y;}
};

template <typename T>
struct greaterequals {
  bool operator() (T x, T y, vm::stack *, size_t=0) {return x >= y;}
};

template <typename T>
struct greater {
  bool operator() (T x, T y, vm::stack *, size_t=0) {return x > y;}
};

template <typename T>
struct notequals {
  bool operator() (T x, T y, vm::stack *, size_t=0) {return x != y;}
};

template <typename T>
struct And {
  bool operator() (T x, T y, vm::stack *, size_t=0) {return x && y;}
};

template <typename T>
struct Or {
  bool operator() (T x, T y, vm::stack *, size_t=0) {return x || y;}
};

template <typename T>
struct Xor {
  bool operator() (T x, T y, vm::stack *, size_t=0) {return x ^ y;}
};

template <typename T>
struct plus {
  T operator() (T x, T y, vm::stack *, size_t=0) {return x+y;}
};

template <typename T>
struct minus {
  T operator() (T x, T y, vm::stack *, size_t=0) {return x-y;}
};
  
template <typename T>
struct times {
  T operator() (T x, T y, vm::stack *, size_t=0) {return x*y;}
};

template <typename T>
struct divide {
  T operator() (T x, T y, vm::stack *s, size_t i=0) {
    if(y == 0) {
      ostringstream buf;
      if(i > 0) buf << "array element " << i << ": ";
      buf << "Divide by zero";
      error(s,buf.str().c_str());
    }
    return x/y;
  }
};

template <typename T>
struct power {
  T operator() (T x, T y, vm::stack *, size_t=0) {return pow(x,y);}
};

template <>
struct power<int> {
  int operator() (int x, int y, vm::stack *s, size_t i=0) {
    if (y < 0 && !(x == 1 || x == -1)) {
      ostringstream buf;
      if(i > 0) buf << "array element " << i << ": ";
      buf << "Only 1 and -1 can be raised to negative exponents as integers.";
      error(s,buf.str().c_str());
    }
    return pow(x,y);
  }
};
 
inline int Mod(int x, int y) {return x % y;}
inline double Mod(double x, double y) {return fmod(x,y);}
  
template<typename T>  
inline T portableMod(T x,T y)
{
// Implementation-independent definition of mod; ensure that result has
// same sign as divisor
  T val=Mod(x,y);
  if((y > 0 && val < 0) || (y < 0 && val > 0)) val += y;
  return val;
}
  
template <typename T>
struct mod {
  T operator() (T x, T y, vm::stack *s, size_t i=0) {
    if(y == 0) {
      ostringstream buf;
      if(i > 0) buf << "array element " << i << ": ";
      buf << "Divide by zero";
      error(s,buf.str().c_str());
    }
    return portableMod(x,y);
  }
};

template <typename T>
struct min {
  T operator() (T x, T y, vm::stack *, size_t=0) {return x < y ? x : y;}
};

template <typename T>
struct max {
  T operator() (T x, T y, vm::stack *, size_t=0) {return x > y ? x : y;}
};

} //namespace trans

namespace run {
  void exitFunction(vm::stack *s);
}

#endif //BUILTIN_H
