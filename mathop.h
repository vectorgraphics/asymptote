/*****
 * mathop.h
 * Tom Prince 2005/3/18
 *
 * Defines some runtime functions used by the stack machine.
 *
 *****/

#ifndef MATHOP_H
#define MATHOP_H

#include <sstream>

#include "stack.h"
#include "pow.h"

namespace run {

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
      std::ostringstream buf;
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
      std::ostringstream buf;
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
      std::ostringstream buf;
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

template <class T>
void Negate(vm::stack *s)
{
  T a=vm::pop<T>(s);
  s->push(-a);
}
  
template <class T, template <class S> class op>
void binaryOp(vm::stack *s)
{
  T b=vm::pop<T>(s);
  T a=vm::pop<T>(s);
  s->push(op<T>()(a,b,s));
}

} // namespace run

#endif MATHOP_H
