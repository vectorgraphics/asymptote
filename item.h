/*****
 * inst.h
 * Tom Prince and John Bowman 2005/04/12
 *
 * Descibes the items that are used by the virtual machine.
 *****/

#ifndef ITEM_H
#define ITEM_H

#include <vector>
#include <gc_allocator.h>
#include <gc_cpp.h>
#include "pool.h"

namespace vm {

class item;
class bad_item_value {};

template<typename T>
T get(const item&);

class item {
public:
  bool empty()
  { return *kind == typeid(void); }
  
  item()
    : kind(&typeid(void)) {}
  
  item(int i)
    : kind(&typeid(int)), i(i) {}
  item(double x)
    : kind(&typeid(double)), x(x) {}
  item(bool b)
    : kind(&typeid(bool)), b(b) {}
  
  item& operator= (int a)
  { kind=&typeid(int); i=a; return *this; }
  item& operator= (double a)
  { kind=&typeid(double); x=a; return *this; }
  item& operator= (bool a)
  { kind=&typeid(bool); b=a; return *this; }
  
  template<class T>
  item(T *p)
    : kind(&typeid(T)), p(p) {}
  
  template<class T>
  item(const T &p)
    : kind(&typeid(T)), p(new T(p)) {}
  
  template<class T>
  item& operator= (T *a)
  { kind=&typeid(T); p=a; return *this; }
  
  template<class T>
  item& operator= (const T &it)
  { kind=&typeid(T); p=new T(it); return *this; }
  
  template<typename T>
  friend inline T get(const item&);

  const std::type_info &type() const
  { return *kind; }
private:
  const std::type_info *kind;
  
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
      if (*it.kind == typeid(T))
	return (T*) it.p;
      throw vm::bad_item_value();
    }
  };
  
  template <typename T>
  struct help {
    static T& unwrap(const item& it)
    {
      if (*it.kind == typeid(T))
	return *(T*) it.p;
      throw vm::bad_item_value();
    }
  };
};
  
class frame : public gc_cleanup {
  typedef std::vector<item,traceable_allocator<item> > vars_t;
  vars_t vars;
public:
  frame(size_t size)
    : vars(size) {}

  item& operator[] (size_t n)
    { return vars[n]; }
  item operator[] (size_t n) const
    { return vars[n]; }

  size_t size()
    { return vars.size(); }
  
  void extend(size_t n)
    { vars.resize(vars.size() + n); }
};

template<typename T>
inline T get(const item& it)
{
  return item::help<T>::unwrap(it);
} 

template <>
inline int get<int>(const item& it)
{
  if (*it.kind == typeid(int))
    return it.i;
  throw vm::bad_item_value();
}
  
template <>
inline double get<double>(const item& it)
{
  if (*it.kind == typeid(double))
    return it.x;
  throw vm::bad_item_value();
}

template <>
inline bool get<bool>(const item& it)
{
  if (*it.kind == typeid(bool))
    return it.b;
  throw vm::bad_item_value();
}

} // namespace vm

#endif // ITEM_H
