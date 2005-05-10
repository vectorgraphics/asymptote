/****
 * memory.h
 *
 * Interface to the Boehm Garbage Collector.
 *****/

#ifndef MEMORY_H
#define MEMORY_H

#include <list>
#include <vector>
#include <deque>
#include <map>

#ifdef USEGC

#define ALLOC gc_allocator

#include <gc.h>

#undef GC_MALLOC
inline void *GC_MALLOC(size_t n) { \
  void *mem=GC_malloc(n); 	   \
  if(mem) return mem; 		   \
  throw std::bad_alloc();	   \
}
  
#include <gc_allocator.h>
#include <gc_cpp.h>

#else // USEGC

#define ALLOC std::allocator
  
class gc {};

enum GCPlacement {UseGC,
#ifndef GC_NAME_CONFLICT
		  GC=UseGC,
#endif
                  NoGC, PointerFreeGC};

extern "C" {typedef void (*GCCleanUpFunc)( void* obj, void* clientData );}

inline void* operator new(size_t size, GCPlacement, GCCleanUpFunc=0) {
  return operator new(size);
}

#endif // USEGC

namespace mem {

#define GC_CONTAINER(KIND)                                              \
  template <typename T>                                                 \
  struct KIND : public std::KIND<T, ALLOC<T> > {                        \
    KIND() : std::KIND<T, ALLOC<T> >() {};                              \
    KIND(size_t n) : std::KIND<T, ALLOC<T> >(n) {};                     \
    KIND(size_t n, const T& t) : std::KIND<T, ALLOC<T> >(n,t) {};       \
  }

GC_CONTAINER(list);
GC_CONTAINER(vector);
GC_CONTAINER(deque);

#undef GC_CONTAINER

#define GC_CONTAINER(KIND)                                              \
  template <typename Key, typename T, typename Compare = std::less<Key> > \
  struct KIND : public std::KIND<Key,T,Compare,ALLOC<std::pair<Key,T> > > { \
    KIND() : std::KIND<Key,T,Compare,ALLOC<std::pair<Key,T> > > () {}; \
  }

GC_CONTAINER(map);
GC_CONTAINER(multimap);

#undef GC_CONTAINER

} // namespace mem

#endif 
