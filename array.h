/*****
 * array.h
 * Tom Prince 2005/06/18
 * 
 * Array type used by virtual machine.
 *****/

#ifndef ARRAY_H
#define ARRAY_H

#include "memory.h"
#include "item.h"

namespace vm {

// Arrays are vectors with a push func for running in asymptote.
class array : public mem::deque<item>, public gc {
public:
  array(size_t n)
    : mem::deque<item>(n)
  {}

  void push(item i)
  {
    push_back(i);
  }

  template <typename T>
  T read(size_t i)
  {
    return get<T>((*this)[i]);
  }
};

template <typename T>
inline T read(array *a, size_t i)
{
  return a->array::read<T>(i);
}

} // namespace vm

#endif // ARRAY_H
