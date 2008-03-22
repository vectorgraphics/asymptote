/*****
 * inst.h
 * Tom Prince and John Bowman 2005/04/12
 *
 * Descibes the items that are used by the virtual machine.
 *****/

#ifndef ITEM_H
#define ITEM_H

#include "common.h"
#include <typeinfo>

namespace vm {

class item;
class bad_item_value {};

template<typename T>
T get(const item&);

class item : public gc {
public:
  bool empty()
  { return *kind == typeid(void); }
  
  item()
    : kind(&typeid(void)) {}
  
#ifndef Int  
  item(Int i)
    : kind(&typeid(Int)), i(i) {}
#endif  
  item(int i)
    : kind(&typeid(Int)), i(i) {}
  item(double x)
    : kind(&typeid(double)), x(x) {}
  item(bool b)
    : kind(&typeid(bool)), b(b) {}
  
#ifndef Int  
  item& operator= (int a)
  { kind=&typeid(Int); i=a; return *this; }
#endif  
  item& operator= (Int a)
  { kind=&typeid(Int); i=a; return *this; }
  item& operator= (double a)
  { kind=&typeid(double); x=a; return *this; }
  item& operator= (bool a)
  { kind=&typeid(bool); b=a; return *this; }
  
  template<class T>
  item(T *p)
    : kind(&typeid(T)), p((void *) p) {}
  
  template<class T>
  item(const T &p)
    : kind(&typeid(T)), p(new(UseGC) T(p)) {}
  
  template<class T>
  item& operator= (T *a)
  { kind=&typeid(T); p=(void *) a; return *this; }
  
  template<class T>
  item& operator= (const T &it)
  { kind=&typeid(T); p=new(UseGC) T(it); return *this; }
  
  template<typename T>
  friend inline T get(const item&);

  friend inline bool isdefault(const item&);
  friend inline bool isarray(const item&);
  
  const std::type_info &type() const
  { return *kind; }
private:
  const std::type_info *kind;
  
  union {
    Int i;
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
  
class frame : public gc {
  typedef mem::vector<item> vars_t;
  vars_t vars;
public:
  frame(size_t size)
    : vars(size)
  {}

  item& operator[] (size_t n)
    { return vars[n]; }
  item operator[] (size_t n) const
    { return vars[n]; }

  size_t size()
    { return vars.size(); }
  
  // Extends vars to ensure it has a place for any variable indexed up to n.
  void extend(size_t n) {
    if (vars.size() < n)
      vars.resize(n);
  }
};

template<typename T>
inline T get(const item& it)
{
  return item::help<T>::unwrap(it);
} 

#ifndef Int  
template <>
inline int get<int>(const item&)
{
  throw vm::bad_item_value();
}
#endif
  
template <>
inline Int get<Int>(const item& it)
{
  if (*it.kind == typeid(Int))
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

// This serves as the object for representing a default argument.
struct default_t : public gc {};
  
inline bool isdefault(const item& it)
{
  return *it.kind == typeid(default_t);
} 

} // namespace vm

GC_DECLARE_PTRFREE(vm::default_t);

#endif // ITEM_H
