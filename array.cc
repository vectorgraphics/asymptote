/*****
 * array.cc
 * Andy Hammerlindl  2008/01/26
 * 
 * Array type used by virtual machine.
 *****/

#include "array.h"
#include "mod.h"

namespace vm {

inline size_t sliceIndex(Int in, size_t len) {
  if (in < 0)
    in += len;
  if (in < 0)
    return 0;
  size_t index = (size_t)in;
  return index < len ? index : len;
}

array *array::slice(Int left, Int right)
{
  size_t length=size();
  if (length == 0)
    return new array();

  if (cycle) {
    if (right <= left)
      return new array();

    size_t resultLength = (size_t)(right - left);
    array *result = new array(resultLength);

    size_t i = (size_t)imod(left, length), ri = 0;
    while (ri < resultLength) {
      (*result)[ri] = (*this)[i];

      ++ri;
      ++i;
      if (i >= length)
        i -= length;
    }

    return result;
  }
  else { // Non-cyclic
    size_t l = sliceIndex(left, length);
    size_t r = sliceIndex(right, length);

    if (r <= l)
      return new array();

    size_t resultLength = r - l;
    array *result = new array(resultLength);

    for (size_t i=0; i<resultLength; ++i)
      (*result)[i] = (*this)[l+i];

    return result;
  }
}

} // namespace vm
