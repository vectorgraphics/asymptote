#ifndef FPU_H
#define FPU_H

#include <fenv.h>

#ifdef _GNU_SOURCE
inline void fpu_trap(bool trap)
{
  if(trap) {// Trap FPU exceptions on NaN, zero divide and overflow.
#ifdef FE_INVALID    
    feenableexcept(FE_INVALID);
#endif    
#ifdef FE_DIVBYZERO
    feenableexcept(FE_DIVBYZERO);
#endif  
#ifdef FE_OVERFLOW
    feenableexcept(FE_OVERFLOW);
#endif  
  } else {// Don't trap FPU exceptions on NaN, zero divide, or overflow.
#ifdef FE_INVALID    
    fedisableexcept(FE_INVALID);
#endif    
#ifdef FE_DIVBYZERO
    fedisableexcept(FE_DIVBYZERO);
#endif  
#ifdef FE_OVERFLOW
    fedisableexcept(FE_OVERFLOW);
#endif    
  }
}
#else
inline void fpu_trap(bool) {}
#endif  

#endif
