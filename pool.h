/****
 * pool.h
 * Tom Prince 2004/07/15
 *
 * Memory tracking utilities.
 *****/

#ifndef POOL_H
#define POOL_H

#include <new>

namespace mempool
{

void free();

class poolitem;
void insert(poolitem);
void erase(poolitem);
  
template <class T>
class pooled
{
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

template <class T>
class poolarray
{
public:
  poolarray() : array(0) {}
  explicit poolarray(T* array);
  explicit poolarray(size_t n);
  void reset (T* new_array);
  T& operator[](std::ptrdiff_t);
  T operator[](std::ptrdiff_t) const;
  bool operator!() const;
  bool operator==(poolarray) const;
  bool operator!=(poolarray) const;
private:
  T* array;
  static void deleter(void*);
};

class poolitem
{
public:
  typedef void (*free_t)(void*);
  poolitem(void *p, free_t free)
    : ptr(p), free_func(free) {}
  void free() const { return free_func(ptr); }
protected:
  friend bool cmp(poolitem,poolitem);
  void* const ptr;
  free_t const free_func;
};

template <class T>
void pooled<T>::deleter(void* ptr)
{
  static_cast<T*>(ptr)->~T();
  ::operator delete (ptr);
}

template <class T>
inline void* pooled<T>::operator new(size_t n)
{
  void *p = ::operator new(n);
  insert(poolitem(p,deleter));
  return p;
}

template <class T>
inline void* pooled<T>::operator new(size_t, void* p)
{
  return p;
}

template <class T>
inline void pooled<T>::operator delete(void* p)
{
  poolitem it(p,deleter);
  erase(it);
  it.free();
}

template <class T>
inline void pooled<T>::operator delete(void*, void*)
{}

template <class T>
void poolarray<T>::deleter(void* ptr)
{
  delete[] static_cast<T*>(ptr);
}

template <class T>
inline poolarray<T>::poolarray(T* array)
  : array(array)
{
  insert(poolitem(array,deleter));
}

template <class T>
inline poolarray<T>::poolarray(size_t n)
  : array(new T[n])
{
  insert(poolitem(array,deleter));
}

template <class T>
inline void poolarray<T>::reset(T* new_array)
{
  array = new_array;
  insert(poolitem(array,deleter));
}

template <class T>
inline T& poolarray<T>::operator[](ptrdiff_t i)
{
  return array[i];
}

template <class T>
inline T poolarray<T>::operator[](ptrdiff_t i) const
{
  return array[i];
}
template <class T>
inline bool poolarray<T>::operator!() const
{
  return (array==0);
}
template <class T>
inline bool poolarray<T>::operator==(poolarray<T> r) const
{
  return (array==r.array);
}
template <class T>
inline bool poolarray<T>::operator!=(poolarray<T> r) const
{
  return (array!=r.array);
}

} // namespace mempool

#endif 
