/****
 * pool.h
 * Tom Prince 2004/07/15
 *
 * Memory tracking utilities.
 *****/

#ifndef POOL_H
#define POOL_H

#include <algorithm>
#include "memory.h"

namespace mem {
  
template <class T>
class managed {
public:
  void *operator new (size_t n);
  static void free();
  managed();
private:
  static void finalizer(void *,void *);
  static void free_it(T *);
  typedef std::list<T*> pool_t;
  typedef typename pool_t::iterator iter_t;
  iter_t ref;
  static pool_t thePool; 
};

template <class T>
std::list<T*> managed<T>::thePool(0);

template <class T>
inline managed<T>::managed()
{
  iter_t iter = thePool.begin();
  while (iter != thePool.end()) {
    if (*iter == this) {
      ref = iter;
      return;
    }
    ++iter;
  }
}

template <class T>
inline void managed<T>::free_it(T *ptr)
  { ptr->~T();
#ifdef USEGC
    GC_REGISTER_FINALIZER(ptr,0,0,0,0);
#endif
    GC_FREE(ptr); }

template <class T>
void managed<T>::free()
{
  std::for_each(thePool.begin(),thePool.end(),free_it);
  pool_t().swap(thePool);
}

template <class T>
void managed<T>::finalizer(void *ptr, void*)
{
  iter_t iter = ((managed<T>*)(T*)ptr)->ref;
  if (iter != iter_t())
    thePool.erase(iter);
  free_it((T*)ptr);
}

template <class T>
inline void* managed<T>::operator new(size_t n)
{
  void *ptr = GC_MALLOC(n);
  thePool.push_front((T*)ptr);
#ifdef USEGC
  GC_REGISTER_FINALIZER_IGNORE_SELF(ptr,&finalizer,0,0,0);
#endif
  return ptr;
}

} // namespace mem

#endif 
