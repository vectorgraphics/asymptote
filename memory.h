/****
 * pool.h
 * Tom Prince 2004/07/15
 *
 * Memory tracking utilities.
 *****/

#ifndef MEMORY_H
#define MEMORY_H

#include <list>
#include <vector>
#include <map>

#include <gc_allocator.h>
#include <gc_cpp.h>
#include "gc_atomic.h"

namespace mem {

#define GC_CONTAINER(KIND)                                              \
  template <typename T>                                                 \
  struct KIND : public std::KIND<T, gc_allocator<T> > {          \
    KIND() : std::KIND<T, gc_allocator<T> >() {};                \
    KIND(size_t n) : std::KIND<T, gc_allocator<T> >(n) {};       \
    KIND(size_t n, const T& t) : std::KIND<T, gc_allocator<T> >(n,t) {}; \
  }
    
GC_CONTAINER(list);
GC_CONTAINER(vector);

#undef GC_CONTAINER

template <typename Key, typename T, typename Compare = std::less<Key> >
struct multimap : public std::multimap<Key,T,Compare,gc_allocator<std::pair<Key,T> > > {
  multimap() : std::multimap<Key,T,Compare,gc_allocator<std::pair<Key,T> > > () {};
};

} // namespace mem

#endif 
