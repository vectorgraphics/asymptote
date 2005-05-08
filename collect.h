#ifndef COLLECT_H
#define COLLECT_H

#ifdef USEGC

#include <gc_allocator.h>
#include <gc_cpp.h>

#else

class gc {};

enum GCPlacement {UseGC,
#ifndef GC_NAME_CONFLICT
		  GC=UseGC,
#endif
                  NoGC, PointerFreeGC};

template<class T>
class gc_allocator : public std::allocator<T> {};

extern "C" {typedef void (*GCCleanUpFunc)( void* obj, void* clientData );}

inline void* operator new(size_t size, GCPlacement, GCCleanUpFunc=0) {
  return new char[size];
}

#endif

#endif
