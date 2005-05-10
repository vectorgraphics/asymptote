/****
 * pool.h
 * Tom Prince 2004/07/15
 *
 * Memory tracking utilities.
 *****/

#ifndef POOL_H
#define POOL_H

#include <deque>
#include <new>

namespace memory {
  
template <class T>
class managed {
public:
  void *operator new (size_t n);
  void operator delete (void*);
// The following are the standard placement new and delete
// operators to make old compilers happy.
  void *operator new (size_t n,void*);
  void operator delete (void*,void*);
private:
  static void deleter(void *);
};

class poolitem {
public:
  typedef void (*free_t)(void*);
  poolitem(void *p, free_t free)
    : ptr(p), free_func(free) {}
  void free() const { return free_func(ptr); }
protected:
  void* ptr;
  free_t free_func;
};

typedef std::deque<poolitem> pool_t;
extern pool_t thePool;
  
inline void free()
{
  for(pool_t::iterator p = thePool.begin(); p != thePool.end(); ++p)
    p->free();
  pool_t().swap(thePool);
}

inline void insert(poolitem p)
{
  thePool.push_back(p);
}

template <class T>
void managed<T>::deleter(void* ptr)
{
  static_cast<T*>(ptr)->~T();
  ::operator delete (ptr);
}

template <class T>
inline void* managed<T>::operator new(size_t n)
{
  void *p = ::operator new(n);
  insert(poolitem(p,deleter));
  return p;
}

template <class T>
inline void* managed<T>::operator new(size_t, void* p)
{
  return p;
}

template <class T>
inline void managed<T>::operator delete(void* p)
{
  poolitem it(p,deleter);
  it.free();
}

template <class T>
inline void managed<T>::operator delete(void*, void*)
{}

} // namespace mempool

#endif 
