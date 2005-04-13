/*****
 * inst.h
 * Tom Prince and John Bowman 2005/04/12
 *
 * Descibes the items that are used by the virtual machine.
 *****/

#ifndef ITEM_H
#define ITEM_H

#include "pool.h"

namespace vm {

class bad_item_value {};

class item {
public:
  bool empty() {return *type == typeid(void);}
  
  item() : type(&typeid(void)) {}
  
  item(int i)
    : type(&typeid(int)), i(i) {}
  item(double x)
    : type(&typeid(double)), x(x) {}
  item(bool b)
    : type(&typeid(bool)), b(b) {}
  
  item& operator= (int a)
  { type=&typeid(int); i=a; return *this; }
  item& operator= (double a)
  { type=&typeid(double); x=a; return *this; }
  item& operator= (bool a)
  { type=&typeid(bool); b=a; return *this; }
  
  template<class T>
  item(T *p)
    : type(&typeid(T)), p(p) {}
  
  template<class T>
  item(const T &p)
    : type(&typeid(T)), p(new T(p)) {}
  
  template<class T>
  item& operator= (T *a)
  { type=&typeid(T); p=a; return *this; }
  
  template<class T>
  item& operator= (const T &it)
  { type=&typeid(T); p=new T(it); return *this; }
  
  template<typename T>
  friend inline T get(const item&);
  
private:
  const std::type_info *type;
  
  union {
    int i;
    double x;
    bool b;
    void *p;
  };

  template <typename T>
  struct help;
  
  template <typename T>
  struct help<T*> {
    static T* unwrap(const item& it)
    {
      if (*it.type == typeid(T))
	return (T*) it.p;
      throw vm::bad_item_value();
    }
  };
  
  template <typename T>
  struct help {
    static T& unwrap(const item& it)
    {
      if (*it.type == typeid(T))
	return *(T*) it.p;
      throw vm::bad_item_value();
    }
  };
};
  
template<typename T>
inline T get(const item& it)
{
  return item::help<T>::unwrap(it);
} 

template <>
inline int get<int>(const item& it)
{
  if (*it.type == typeid(int))
    return it.i;
  throw vm::bad_item_value();
}
  
template <>
inline double get<double>(const item& it)
{
  if (*it.type == typeid(double))
    return it.x;
  throw vm::bad_item_value();
}

template <>
inline bool get<bool>(const item& it)
{
  if (*it.type == typeid(bool))
    return it.b;
  throw vm::bad_item_value();
}

typedef memory::managed_array<item> frame;

} // namespace vm

#endif // ITEM_H
