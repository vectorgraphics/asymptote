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
#include "mod.h"
#include "pow.h"

namespace run {

template <typename T>
struct less {
  bool operator() (T x, T y, size_t=0) {return x < y;}
};

template <typename T>
struct lessequals {
  bool operator() (T x, T y, size_t=0) {return x <= y;}
};

template <typename T>
struct equals {
  bool operator() (T x, T y, size_t=0) {return x == y;}
};

template <typename T>
struct greaterequals {
  bool operator() (T x, T y, size_t=0) {return x >= y;}
};

template <typename T>
struct greater {
  bool operator() (T x, T y, size_t=0) {return x > y;}
};

template <typename T>
struct notequals {
  bool operator() (T x, T y, size_t=0) {return x != y;}
};

template <typename T>
struct And {
  bool operator() (T x, T y, size_t=0) {return x && y;}
};

template <typename T>
struct Or {
  bool operator() (T x, T y, size_t=0) {return x || y;}
};

template <typename T>
struct Xor {
  bool operator() (T x, T y, size_t=0) {return x ^ y;}
};

template <typename T>
struct plus {
  T operator() (T x, T y, size_t=0) {return x+y;}
};

template <typename T>
struct minus {
  T operator() (T x, T y, size_t=0) {return x-y;}
};
  
template <typename T>
struct times {
  T operator() (T x, T y, size_t=0) {return x*y;}
};

extern void dividebyzero(size_t i=0);  
  
template <typename T>
struct divide {
  T operator() (T x, T y,  size_t i=0) {
    if(y == 0) dividebyzero(i);
    return x/y;
  }
};

template<>
struct divide<int> {
  double operator() (int x, int y,  size_t i=0) {
    if(y == 0) dividebyzero(i);
    return ((double) x)/(double) y;
  }
};

template <typename T>
struct power {
  T operator() (T x, T y, size_t=0) {return pow(x,y);}
};

template <>
struct power<int> {
  int operator() (int x, int y,  size_t i=0) {
    if (y < 0 && !(x == 1 || x == -1)) {
      std::ostringstream buf;
      if(i > 0) buf << "array element " << i << ": ";
      buf << "Only 1 and -1 can be raised to negative exponents as integers.";
      vm::error(buf.str().c_str());
    }
    return pow(x,y);
  }
};
 
template <typename T>
struct mod {
  T operator() (T x, T y,  size_t i=0) {
    if(y == 0) dividebyzero(i);
    return portableMod(x,y);
  }
};

template <typename T>
struct min {
  T operator() (T x, T y, size_t=0) {return x < y ? x : y;}
};

template <typename T>
struct max {
  T operator() (T x, T y, size_t=0) {return x > y ? x : y;}
};

template <class T>
void Negate(vm::stack *s)
{
  T a=vm::pop<T>(s);
  s->push(-a);
}

template <double (*func)(double)>
void realReal(vm::stack *s) 
{
  s->push(func(vm::pop<double>(s)));
}

template <class T, template <class S> class op>
void binaryOp(vm::stack *s)
{
  T b=vm::pop<T>(s);
  T a=vm::pop<T>(s);
  s->push(op<T>()(a,b));
}

} // namespace run

#endif //MATHOP_H

