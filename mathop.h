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
extern void integeroverflow(size_t i=0);  
  
template <typename T>
struct divide {
  T operator() (T x, T y,  size_t i=0) {
    if(y == 0) dividebyzero(i);
    return x/y;
  }
};

inline bool validInt(double x) {
  return x > Int_MIN-0.5 && x < Int_MAX+0.5;
}
  
inline void checkInt(double x, size_t i)
{
  if(validInt(x)) return;
  integeroverflow(i);
}
  
inline Int Intcast(double x)
{
  if(validInt(x)) return (Int) x;
  integeroverflow(0);
  return 0;
}
  
template<>
struct plus<Int> {
  Int operator() (Int x, Int y, size_t i=0) {
    if((y > 0 && x > Int_MAX-y) || (y < 0 && x < Int_MIN-y))
      integeroverflow(i);
    return x+y;
  }
};

template<>
struct minus<Int> {
  Int operator() (Int x, Int y, size_t i=0) {
    if((y < 0 && x > Int_MAX+y) || (y > 0 && x < Int_MIN+y))
      integeroverflow(i);
    return x-y;
  }
};

template<>
struct times<Int> {
  Int operator() (Int x, Int y, size_t i=0) {
    if(y == 0) return 0;
    if(y < 0) {y=-y; x=-x;}
    if(x > Int_MAX/y || x < Int_MIN/y)
       integeroverflow(i);
    return x*y;
  }
};

template<>
struct divide<Int> {
  double operator() (Int x, Int y, size_t i=0) {
    if(y == 0) dividebyzero(i);
    return ((double) x)/(double) y;
  }
};

template <class T>
void Negate(vm::stack *s)
{
  T a=vm::pop<T>(s);
  s->push(-a);
}

template<>
inline void Negate<Int>(vm::stack *s)
{
  Int a=vm::pop<Int>(s);
  if(a < -Int_MAX) integeroverflow(0);
  s->push(-a);
}

template <typename T>
struct power {
  T operator() (T x, T y, size_t=0) {return pow(x,y);}
};

template <>
struct power<Int> {
  Int operator() (Int x, Int p,  size_t i=0) {
    if(p == 0) return 1;
    Int sign=1;
    if(x < 0) {
      if(p % 2) sign=-1;
      x=-x;
    }
    if(p > 0) {
      if(x == 0) return 0;
      Int r = 1;
      for(;;) {
	if(p & 1) {
	  if(r > Int_MAX/x) integeroverflow(i);
	  r *= x;
	}
	if((p >>= 1) == 0)
	  return sign*r;
	if(x > Int_MAX/x) integeroverflow(i);
	x *= x;
      }
    } else {
      if(x == 1) return sign;
      ostringstream buf;
      if(i > 0) buf << "array element " << i << ": ";
      buf << "Only 1 and -1 can be raised to negative exponents as integers.";
      vm::error(buf);
      return 0;
    }
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

template <double (*func)(double)>
void realReal(vm::stack *s) 
{
  double x=vm::pop<double>(s);
  s->push(func(x));
}

template <class T, template <class S> class op>
void binaryOp(vm::stack *s)
{
  T b=vm::pop<T>(s);
  T a=vm::pop<T>(s);
  s->push(op<T>()(a,b));
}

template <class T>
void interp(vm::stack *s)
{
  double t=vm::pop<double>(s);
  T b=vm::pop<T>(s);
  T a=vm::pop<T>(s);
  s->push(a+t*(b-a));
}

} // namespace run

#endif //MATHOP_H

