/*****
 * mod.h
 * Andy Hammerlindl 2005/03/16
 *
 * Definition of implementation independent mod function.
 *****/

#ifndef MOD_H
#define MOD_H

#include <cmath>
using std::fmod;

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
  
inline int imod(int x, int y)
{
  return portableMod<int>(x,y);
}

inline int imod(int i, unsigned int n) {
  i %= (int) n;
  if(i < 0) i += (int) n;
  return i;
}

#endif
