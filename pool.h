/****
 * pool.h
 * Tom Prince 2004/07/15
 *
 * Memory tracking utilities.
 *****/

#ifndef POOL_H
#define POOL_H

#include <new>

namespace memory
{

void free();

class poolitem;
void insert(poolitem);
void erase(poolitem);
  
template <class T>
class managed
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
class managed_array
{
public:
  managed_array() : array(0) {}
  explicit managed_array(T* array);
  explicit managed_array(size_t n);
  void reset (T* new_array);
  T& operator[](std::ptrdiff_t);
  T operator[](std::ptrdiff_t) const;
  bool operator!() const;
  bool operator==(managed_array) const;
  bool operator!=(managed_array) const;
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
  void* ptr;
  free_t free_func;
};

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
  erase(it);
  it.free();
}

template <class T>
inline void managed<T>::operator delete(void*, void*)
{}

template <class T>
void managed_array<T>::deleter(void* ptr)
{
  delete[] static_cast<T*>(ptr);
}

template <class T>
inline managed_array<T>::managed_array(T* array)
  : array(array)
{
  insert(poolitem(array,deleter));
}

template <class T>
inline managed_array<T>::managed_array(size_t n)
  : array(new T[n])
{
  insert(poolitem(array,deleter));
}

template <class T>
inline void managed_array<T>::reset(T* new_array)
{
  array = new_array;
  insert(poolitem(array,deleter));
}

template <class T>
inline T& managed_array<T>::operator[](ptrdiff_t i)
{
  return array[i];
}

template <class T>
inline T managed_array<T>::operator[](ptrdiff_t i) const
{
  return array[i];
}
template <class T>
inline bool managed_array<T>::operator!() const
{
  return (array==0);
}
template <class T>
inline bool managed_array<T>::operator==(managed_array<T> r) const
{
  return (array==r.array);
}
template <class T>
inline bool managed_array<T>::operator!=(managed_array<T> r) const
{
  return (array!=r.array);
}

} // namespace mempool

#endif 
