/*****
 * inst.h
 * Tom Prince 2005/03/20
 * 
 * Descibes the items that are used by the virtual machine.
 *****/

#ifndef ITEM_H
#define ITEM_H

namespace vm {

typedef boost::any item;
typedef memory::managed_array<item> frame;

template<typename T>
inline T get(const item& val)
{
  return boost::any_cast<T>(val);
} 

} // namespace vm

#endif // ITEM_H
