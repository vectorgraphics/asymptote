/*****
 * table.h
 * Andy Hammerlindl 2002/06/18
 *
 * Table used to bind symbols to vars and types in a namespace.
 *****/

#ifndef TABLE_H
#define TABLE_H

#include <cassert>
#include <map>
#include <list>
#include <utility>

#include "symbol.h"
#include "memory.h"

namespace sym {

template <class B>
class table;

template <class B>
std::ostream& operator<< (std::ostream& out, const table<B>& t);

#if defined(__DECCXX_LIBCXX_RH70)
#define CONST
#else
#define CONST const  
#endif
  
template <class B>
class table {
protected:
  typedef mem::multimap<symbol*CONST,B> scope_t;
  typedef typename scope_t::iterator scope_iterator;
  typedef mem::list<scope_t> scopes_t;
  typedef mem::list<B> name_t;
  typedef typename name_t::iterator name_iterator;
  typedef mem::map<symbol*CONST,name_t> names_t;

  scopes_t scopes;
  names_t names;

  void remove(symbol *key);
public :
  table();

  void enter(symbol *key, B value);
  B look(symbol *key);

  // Checks if a symbol was added in the last scope.  Useful for
  // stopping multiple definitions.
  B lookInTopScope(symbol *key);

  // Allows scoping and overloading of symbols of the same name
  void beginScope();
  void endScope();

  friend std::ostream& operator<< <B> (std::ostream& out, const table& t);
};

template <class B>
inline table<B>::table()
{
  beginScope();
}

template <class B>
inline void table<B>::enter(symbol *key, B value)
{
  scopes.front().insert(std::make_pair(key,value));
  names[key].push_front(value);
}

template <class B>
inline B table<B>::look(symbol *key)
{
  if (!names[key].empty())
    return names[key].front();
  return 0;
}

template <class B>
inline B table<B>::lookInTopScope(symbol *key)
{
  // Due to the structure of the hash, this lookup has to be done by a linear
  // search.  However, the top scope is usually fairly empty, so it is not a
  // problem.
  scope_iterator p = scopes.front().find(key);
  if (p!=scopes.front().end())
    return p->second;
  return 0;
}

template <class B>
inline void table<B>::beginScope()
{
  scopes.push_front(scope_t());
}

template <class B>
inline void table<B>::remove(symbol *key)
{
  if (!names[key].empty())
    names[key].pop_front();
}

template <class B>
inline void table<B>::endScope()
{
  scope_t &scope = scopes.front();
  for (scope_iterator p = scope.begin(); p != scope.end(); ++p)
    remove(p->first);
  scopes.pop_front();
}

} // namespace sym

#endif
