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
#include <stack>
#include <map>
#include <string>
#include <sstream>

#ifndef NOHASH
#include <ext/hash_map>
#endif

#if defined(__DECCXX_LIBCXX_RH70)
#define CONST
#else
#define CONST const  
#endif
  
#ifdef USEGC

#include <gc.h>

extern "C" {
#include <gc_backptr.h>
}

#undef GC_MALLOC
# ifdef GC_DEBUG
inline void *GC_MALLOC(size_t n) { \
  if (void *mem=GC_debug_malloc(n, GC_EXTRAS))	\
    return mem;                    \
  GC_generate_random_backtrace();  \
  throw std::bad_alloc();          \
}
#else  
inline void *GC_MALLOC(size_t n) { \
  if (void *mem=GC_malloc(n))  \
    return mem;                    \
  throw std::bad_alloc();          \
}
#endif

#include <gc_allocator.h>
#include <gc_cpp.h>

#else // USEGC

using std::allocator;
#define gc_allocator allocator

class gc {};
class gc_cleanup {};

enum GCPlacement {UseGC, NoGC, PointerFreeGC};

inline void* operator new(size_t size, GCPlacement) {
  return operator new(size);
}

#define GC_MALLOC(size) ::operator new(size)
#define GC_FREE(ptr) ::operator delete(ptr)

#endif // USEGC

namespace mem {

#define GC_CONTAINER(KIND)                                               \
  template <typename T>                                                  \
  struct KIND : public std::KIND<T, gc_allocator<T> > {                  \
    KIND() : std::KIND<T, gc_allocator<T> >() {}                         \
    KIND(size_t n) : std::KIND<T, gc_allocator<T> >(n) {}                \
    KIND(size_t n, const T& t) : std::KIND<T, gc_allocator<T> >(n,t) {}  \
  }

GC_CONTAINER(list);
GC_CONTAINER(vector);
GC_CONTAINER(deque);

template <typename T, typename Container = deque<T> >
struct stack : public std::stack<T, Container> {
};

#undef GC_CONTAINER

#define GC_CONTAINER(KIND)                                                    \
  template <typename Key, typename T, typename Compare = std::less<Key> >     \
  struct KIND : public                                                        \
  std::KIND<Key,T,Compare,gc_allocator<std::pair<Key,T> > > {                 \
    KIND() : std::KIND<Key,T,Compare,gc_allocator<std::pair<Key,T> > > () {}  \
  }

GC_CONTAINER(map);
GC_CONTAINER(multimap);

#undef GC_CONTAINER

#ifndef NOHASH
#define EXT __gnu_cxx
#define GC_CONTAINER(KIND)                                                    \
  template <typename Key, typename T,                                         \
            typename Hash = EXT::hash<Key>,                                   \
            typename Eq = std::equal_to<Key> >                                \
  struct KIND : public                                                        \
  EXT::KIND<Key,T,Hash,Eq,gc_allocator<std::pair<Key, T> > > {                \
    KIND() : EXT::KIND<Key,T,Hash,Eq,gc_allocator<std::pair<Key, T> > > () {} \
  }

GC_CONTAINER(hash_map);
GC_CONTAINER(hash_multimap);

#undef GC_CONTAINER
#undef EXT
#endif

#ifdef USEGC
typedef std::basic_string<char,std::char_traits<char>,gc_allocator<char> > string;
typedef std::basic_istringstream<char,std::char_traits<char>,gc_allocator<char> > istringstream;
typedef std::basic_ostringstream<char,std::char_traits<char>,gc_allocator<char> > ostringstream;
typedef std::basic_stringbuf<char,std::char_traits<char>,gc_allocator<char> > stringbuf;
#else
typedef std::string string;
typedef std::istringstream istringstream;
typedef std::ostringstream ostringstream;
typedef std::stringbuf stringbuf;
#endif // USEGC


} // namespace mem

#endif 
